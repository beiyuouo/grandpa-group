#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" 
@File    :   main.py 
@Time    :   2022-01-17 19:48:59 
@Author  :   Bingjie Yan 
@Email   :   bj.yan.pa@qq.com 
@License :   Apache License 2.0 
"""

import os
from argparse import ArgumentParser
from time import sleep

import cv2

from api.pose_detect import PoseDetector
from api.scence import Scence

parser = ArgumentParser()
parser.add_argument("-v",
                    "--video",
                    default=os.path.join('data', 'video.mp4'),
                    help="video path")
parser.add_argument("-o", "--output", default='output', help="output directory")
parser.add_argument("--fps", default=None, type=int, help="fps")
parser.add_argument("-bg",
                    "--background",
                    default=os.path.join('data', 'bg.png'),
                    help="background image path")
parser.add_argument('--clear', action='store_true', help='do not clear background')
parser.add_argument('--debug', action='store_true')


def prepare(args):
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'temp'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'temp_gg'), exist_ok=True)
    if args.clear:
        for f in os.listdir(os.path.join(args.output, 'temp')):
            os.remove(os.path.join(args.output, 'temp', f))

        for f in os.listdir(os.path.join(args.output, 'temp_gg')):
            os.remove(os.path.join(args.output, 'temp_gg', f))


def run(args):
    sc = Scence(args)

    cap = cv2.VideoCapture(args.video)

    args.fps = cap.get(cv2.CAP_PROP_FPS)

    detector = PoseDetector()

    itr = 0

    while True:
        success, img = cap.read()
        if not success:
            break
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False)

        # print(lmList)
        # print(bboxInfo)

        img_sc = sc.get_scence(lmList)

        cv2.imshow('img', img)
        cv2.imshow("sc", img_sc)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # sleep(1)
        sc.save_scence(os.path.join(args.output, 'temp', f'{itr:06d}.png'))
        cv2.imwrite(os.path.join(args.output, 'temp_gg', f'{itr:06d}.png'), img)
        itr += 1

    cap.release()
    cv2.destroyAllWindows()


def post_process(args):
    # merge
    os.system(
        f'ffmpeg -f image2 -r {args.fps} -i {args.output}/temp/%06d.png -vcodec libx264 -crf 25 -r {args.fps} -pix_fmt yuv420p {args.output}/output.mp4 -y'
    )
    os.system(
        f'ffmpeg -f image2 -r {args.fps} -i {args.output}/temp_gg/%06d.png -vcodec libx264 -crf 25 -r {args.fps} -pix_fmt yuv420p {args.output}/output_gg.mp4 -y'
    )

    # output with music in args.video
    os.system(
        f'ffmpeg -i {args.video} -i {args.output}/output.mp4 -map 0:a -map 1:v -c:v copy -c:a copy {args.output}/output_with_music.mp4 -y'
    )

    # merge source video and output video
    os.system(
        f'ffmpeg -i {args.video} -i {args.output}/output_gg.mp4 -i {args.output}/output_with_music.mp4 -filter_complex "[0:v]pad=iw*3:ih*1[a];[a][1:v]overlay=w[b];[b][2:v]overlay=2.0*w" {args.output}/triple_video.mp4 -y'
    )

    # sample gif
    os.system(
        f'ffmpeg -i {args.output}/triple_video.mp4 -vf fps=10,scale=320:-1 -t 00:00:10.000 {args.output}/triple_video.gif -y'
    )

    # remove temp files
    if args.clear:
        for f in os.listdir(os.path.join(args.output, 'temp')):
            os.remove(os.path.join(args.output, 'temp', f))

        for f in os.listdir(os.path.join(args.output, 'temp_gg')):
            os.remove(os.path.join(args.output, 'temp_gg', f))


def main():
    args = parser.parse_args()
    if args.fps is None:
        args.fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS)
    prepare(args)
    run(args)
    post_process(args)


if __name__ == '__main__':
    main()
