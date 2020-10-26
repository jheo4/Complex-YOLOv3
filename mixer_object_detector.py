import numpy as np
import math
import os
import argparse
import cv2
import time
import torch
import zmq
import struct

import utils.utils as utils
from models import *
import torch.utils.data as torch_data

import utils.kitti_utils as kitti_utils
import utils.kitti_aug_utils as aug_utils
import utils.kitti_bev_utils as bev_utils
from utils.kitti_yolo_dataset import KittiYOLODataset
import utils.config as cnf
import utils.mayavi_viewer as mview
from utils.mixer_utils import ObjectContext
from utils.mixer_utils import DetectorOptions

def predictions_to_kitti_format(img_detections, calib, img_shape_2d, img_size, RGB_Map=None):
    # Max Objects: 50
    predictions = np.zeros([50, 7], dtype=np.float32)
    count = 0

    # How to interpret the detected objects in BEV
    for detections in img_detections:
        if detections is None:
            continue
        # Rescale the detection info w.r.t. the BEV image
        for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
            yaw = np.arctan2(im, re)
            predictions[count, :] = cls_pred, x/img_size, y/img_size, w/img_size, l/img_size, im, re
            count += 1

    # inverse X-Y, adjust predictions
    predictions = bev_utils.inverse_yolo_target(predictions, cnf.boundary)

    # Convert LiDAR prediction values into camera values: x, y, z, h, w, l, ry(rx->ry)
    if predictions.shape[0]:
        predictions[:, 1:] = aug_utils.lidar_to_camera_box(predictions[:, 1:], calib.V2C, calib.R0, calib.P)

    objects_new = []
    object_contexts = []
    for index, l in enumerate(predictions):
        str = "Pedestrian"
        if l[0] == 0:str="Car"
        elif l[0] == 1:str="Pedestrian"
        elif l[0] == 2: str="Cyclist"
        else:str = "DontCare"
        line = '%s -1 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0' % str

        # Set object details
        #   - t: location (x, y, z) in camera coord
        #   - h, w, l: box width, height, length
        #   - ry: yaw in camera coord (-pi ~ pi)
        obj = kitti_utils.Object3d(line)
        obj.t = l[1:4]
        obj.h, obj.w, obj.l = l[4:7]
        obj.ry = np.arctan2(math.sin(l[7]), math.cos(l[7]))
        print(f"type: {obj.type}, classid: {obj.cls_id}, truncation: {obj.truncation}, occlusion: {obj.occlusion} ",
                f"alpha: {obj.alpha}, x: {obj.xmin}, y: {obj.ymin} \n",
                f"boxHeight: {obj.h}, boxWidth: {obj.w}, boxLen: {obj.l}, locatinoInCamCoord: {obj.t}",
                f"distToCam: {obj.t}, yawInCamCoord: {obj.ry}\n")


        # Get the 3D box corners with width, height, length. Then, rotate the corners to y-axis (ry)
        #_, corners_3d = kitti_utils.compute_box_3d(obj, calib.P)
        xyz, ry = kitti_utils.compute_3d_object_context(obj, calib.P)
        obj_ctx = ObjectContext()
        obj_ctx.xyz = xyz
        obj_ctx.ry = ry
        obj_ctx.cls_pred = obj.cls_id

        objects_new.append(obj)
        object_contexts.append(obj_ctx)

    if RGB_Map is not None:
        labels, noObjectLabels = kitti_utils.read_labels_for_bevbox(objects_new)
        if not noObjectLabels:
            labels[:, 1:] = aug_utils.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P)
        target = bev_utils.build_yolo_target(labels)
        utils.draw_box_in_bev(RGB_Map, target)

    return objects_new, object_contexts


if __name__ == "__main__":
    opt = DetectorOptions()

    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", type=str, default="localhost", help="the address of the destination")
    parser.add_argument("--port", type=str, default="5555", help="the binding port of the destination")
    args = parser.parse_args()

    '''
        Load DL model: ComplexYOLOv3
    '''
    classes = utils.load_classes(opt.class_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))
    # Eval mode
    model.eval()

    '''
        Load Dataset: KITTI Test
    '''
    dataset = KittiYOLODataset(cnf.root_dir, split=opt.split, mode='TEST', folder=opt.folder, data_aug=False)
    data_loader = torch_data.DataLoader(dataset, 1, shuffle=False)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    '''
        Setup ZMQ socket
    '''
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    addr = "tcp://" + args.addr + ":" + args.port
    print(f"connect to {addr}")
    socket.connect(addr)

    start_time = time.time()
    for index, (img_paths, bev_maps) in enumerate(data_loader):
        # Get detections with bev input image
        input_imgs = Variable(bev_maps.type(Tensor))
        with torch.no_grad():
            detections = model(input_imgs)
            detections = utils.non_max_suppression_rotated_bbox(detections, opt.conf_thres, opt.nms_thres)

        end_time = time.time()
        print(f"FPS: {(1.0/(end_time-start_time)):0.2f}")
        start_time = end_time

        # Read the P2 camera frame corresponding to BEV img
        print(img_paths[0])
        img2d = cv2.imread(img_paths[0])

        # Load calibration
        #   - P0~P3: projection matricies for 4 Camera (P2 is used)
        #   - R0: rotation from reference camera to rectified camera coordinates
        #   - Tr_velo_to_cam: translation from velodyne to camera
        #   - Tr_imu_to_velo: translation from imu to velodyne
        calib = kitti_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))

        # convert the BEV predictions into KITTI format
        objects_pred, object_contexts = predictions_to_kitti_format(detections, calib, img2d.shape, opt.img_size)

        # send object_contexts to AR pipeline
        packed_obj_num = struct.pack('1i', len(object_contexts))
        socket.send(packed_obj_num)
        socket.recv()

        for object_context in object_contexts:
            xyzrycls = object_context.xyz
            xyzrycls = np.append(xyzrycls, object_context.ry)
            xyzrycls = np.append(xyzrycls, object_context.cls_pred)
            print(xyzrycls)
            packed_xyzrycls = struct.pack('5f', *xyzrycls)
            socket.send(packed_xyzrycls)
            socket.recv()


        img2d = mview.show_image_with_boxes(img2d, objects_pred, calib, False)
        cv2.imshow("img2d", img2d)

        bev_maps = torch.squeeze(bev_maps).numpy()
        rgb_map = np.zeros((cnf.BEV_WIDTH, cnf.BEV_WIDTH, 3))
        rgb_map[:, :, 2] = bev_maps[0, :, :]
        rgb_map[:, :, 1] = bev_maps[1, :, :]
        rgb_map[:, :, 0] = bev_maps[2, :, :]
        rgb_map *= 255
        rgb_map = rgb_map.astype(np.uint8)

        img_detections = []
        img_detections.extend(detections)

        for detections in img_detections:
            if detections is None:
                continue

            # Rescale boxes to original image
            detections = utils.rescale_boxes(detections, opt.img_size, rgb_map.shape[:2])
            for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
                yaw = np.arctan2(im, re)
                # Draw rotated box
                bev_utils.drawRotatedBox(rgb_map, x, y, w, l, yaw, cnf.colors[int(cls_pred)])
        cv2.imshow("bev img", rgb_map)

        if cv2.waitKey(0) & 0xFF == 27:
            break

