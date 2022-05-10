from __future__ import annotations
import xml.etree.ElementTree as ET
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-a07', '--annotations-2007', dest='annotations_2007',
    help='path to the PASCAL VOC 2007 annotations directory, i.e \
          Annotations folder',
    default='VOCdevkit/VOC2007/Annotations/'
)
parser.add_argument(
    '-a12', '--annotations-2012', dest='annotations_2012',
    help='path to the PASCAL VOC 2012 annotations directory, i.e \
          Annotations folder',
    default='VOCdevkit/VOC2012/Annotations/'
)
args = vars(parser.parse_args())

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        difficult = int(obj.find('difficult').text)
        if difficult == 1:
            continue
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)

    return objects

file_sets = [
    ('train.txt', 'train_labels.txt'),
    ('2007_test.txt', '2007_test_labels.txt')
]

for i, files in enumerate(file_sets):
    print(files)
    txt_file = open(files[1],'w')
    test_file = open(files[0],'r')
    lines = test_file.readlines()
    lines = [x[:-1] for x in lines]

    if files[0] == 'train.txt':
        xml_files = os.listdir(args['annotations_2012'])
        xml_files.extend(os.listdir(args['annotations_2007']))
    else:
        xml_files = os.listdir(args['annotations_2007'])

    count = 0
    for line in lines:
        count += 1
        image_path = line
        print(image_path)
        xml_file = line.split(os.path.sep)[-1].split('.')[0] + '.xml'
        print(xml_file)
        results = parse_rec(
            '/'.join(image_path.split(os.path.sep)[:-2]) + \
            '/Annotations/' + xml_file
        )
        if len(results)==0:
            print(xml_file)
            continue
        txt_file.write(image_path)
        for result in results:
            class_name = result['name']
            bbox = result['bbox']
            class_name = VOC_CLASSES.index(class_name)
            txt_file.write(' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(class_name))
        txt_file.write('\n')
    txt_file.close()