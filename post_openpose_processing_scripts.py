import os
import re
import pickle
import json
from MinimumBoundingBox import MinimumBoundingBox
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import math
import sys
from scipy.stats import linregress
from operator import itemgetter
import scipy.misc
import xml.etree.ElementTree
from itertools import groupby
import string
from shutil import copyfile
from random import shuffle
import csv

openpose_keypoints = ["radius_x","radius_y","radius_c","scaphoid_x","scaphoid_y","scaphoid_c","thumb_trapezium_x","thumb_trapezium_y","thumb_trapezium_c","thumb_metacarpal_x","thumb_metacarpal_y","thumb_metacarpal_c","thumb_phalange_x","thumb_phalange_y","thumb_phalange_c","index_trapezium_x","index_trapezium_y","index_trapezium_c","index_metacarpal_x","index_metacarpal_y","index_metacarpal_c","index_proximal_x","index_proximal_y","index_proximal_c","index_phalange_x","index_phalange_y","index_phalange_c","middle_trapezium_x","middle_trapezium_y","middle_trapezium_c","middle_metacarpal_x","middle_metacarpal_y","middle_metacarpal_c","middle_proximal_x","middle_proximal_y","middle_proximal_c","middle_phalange_x","middle_phalange_y","middle_phalange_c","ring_trapezium_x","ring_trapezium_y","ring_trapezium_c","ring_metacarpal_x","ring_metacarpal_y","ring_metacarpal_c","ring_proximal_x","ring_proximal_y","ring_proximal_c","ring_phalange_x","ring_phalange_y","ring_phalange_c","little_trapezium_x","little_trapezium_y","little_trapezium_c","little_metacarpal_x","little_metacarpal_y","little_metacarpal_c","little_proximal_x","little_proximal_y","little_proximal_c","little_phalange_x","little_phalange_y","little_phalange_c"]

openpose_keypoints_connections = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16), (0,17), (17,18), (18,19), (19,20)]

both_hands_signtype = ["2-h\xc3\xa5nd paralel", "2-h\xc3\xa5nd spejlsymetrisk", "2-h\xc3\xa5nd punktsymetrisk"]

one_hand_signtype = ["1-h\xc3\xa5nd", "1-h\xc3\xa5nd H2"]

window_size = int(sys.argv[1]) if len(sys.argv) > 1 else 3
print "window_size", window_size

graphs_per_page = 9 #must be sqrtable
bucket_size = 10
total_buckets = 10
frame_size_wh = (720,576)
exec_option=7

'''
1 - collect all json files from the openpose, sort them after the frame number and save as speeds_barchart.pickle
2 - calculate cut-off points by calculating distances travelled between the frames using the non-overlapping window of minimum size (
3) and calculating the slopes and cutting like /...\ and then saving the cut-off points in speeds_barchart_cutoff_points.pickle
3 - saving resulting dataset on the filesystem with format folder/file/right|left/.png
4 - looking up classes using the xml file and saving the new dataset with format class/folder_file_right|left.png
5 - splitting dataset into training/validation/test
6 - extract json ds with classes
'''

def csv_file_contains(filename, astring):
    #print "cvs_file_contains", filename
    found_row = False
    if os.path.isfile(filename):
        csv_file = csv.reader(open(filename, "rb"), delimiter=",")
        for row in csv_file:
            #print row, "first el", row[0], "searching for", astring
            #if current rows 2nd value is equal to input, print that row
            if astring == row[0]:
                found_row = True

    return found_row


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def get_line_between_2_points(x,y, steps):
    coefficients = np.polyfit(x, y, 1)

    # Let's compute the values of the line...
    polynomial = np.poly1d(coefficients)
    x_axis = np.linspace(x[0],x[1], steps)
    y_axis = polynomial(x_axis)

    return zip(x_axis, y_axis)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def autolabel(rects, frames, folder_idx):
    """
    Attach a text label above each bar displaying its height
    """
    for idx, rect in enumerate(rects):
        height = rect.get_height()
        ax[folder_idx / int(math.sqrt(graphs_per_page)), folder_idx % int(math.sqrt(graphs_per_page))].text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%s' % '\n'.join(frames[idx]),
                ha='center', va='bottom', fontsize=6)

def centeroidpython(data):
    k=3   
    del data[k-1::k]

    data = zip(data[0::2], data[1::2])

    #print data
    
    x, y = zip(*data)
    l = len(x)
    return sum(x) / l, sum(y) / l

def sorted_nicely( l ):
    """ Sorts the given iterable in the way that is expected.
 
    Required arguments:
    l -- The iterable to be sorted.
 
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)


if not os.path.isfile("speeds_barchart.pickle") and exec_option==1:
    dir_files_dict = {}
    #list_of_files = []
    last_dirpath = ""
    for (dirpath, dirnames, filenames) in os.walk("/home/bmocialov/tegnsprag_frames_openpose_json"):
        list_of_files = []
        for filename in filenames:
            if filename.endswith('.json'):
                #if dirpath != last_dirpath:
                #    list_of_files = []
                #    last_dirpath = dirpath
                list_of_files.append(filename)
                #print filename, dirpath
        dir_files_dict[dirpath] = sorted_nicely(list_of_files)

    with open('speeds_barchart.pickle', 'wb') as handle:
        pickle.dump(dir_files_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

cut_off_points = {}
if os.path.isfile("speeds_barchart.pickle") and exec_option==2:
    with open("speeds_barchart.pickle", "rb") as input_file:
        dir_files_dict = pickle.load(input_file)

    graph_index = 0
    folders_visited_titles = []
    for folder_idx, a_folder in enumerate(dir_files_dict):
        #if folder_idx > graphs_per_page+1: continue #temporary

        both_hands_distances_traveled_per_sign = []
        #print "folder", a_folder
        #cut_off_points[a_folder.split("/")[-1]] = {}
        last_point = None
        for idx, a_file in enumerate(dir_files_dict[a_folder][::window_size]): #remove the step size if want overlapping
            #print "landed on file", a_file
            right_hand_window_centroids = []
            left_hand_window_centroids = []
            if int(dir_files_dict[a_folder][-1].split("-")[1].split("_")[0]) >= int(a_file.split("-")[1].split("_")[0])+window_size:
                for window_idx in range(0, window_size):
                    with open(a_folder+"/"+a_file.replace(a_file.split("-")[1].split("_")[0], str(int(a_file.split("-")[1].split("_")
[0])+window_idx))) as f:
                        #print "opening file", a_folder+"/"+a_file.replace(a_file.split("-")[1].split("_")[0], str(int(a_file.split("-")[1].split("_")[0])+window_idx))
                        data = json.load(f)
                        if centeroidpython(data["people"][0]["hand_right_keypoints_2d"]).count(centeroidpython(data["people"][0]["hand_right_keypoints_2d"])[0]) != len(centeroidpython(data["people"][0]["hand_right_keypoints_2d"])) and centeroidpython(data["people"][0]["hand_right_keypoints_2d"])[0] != 0:
                            right_hand_centroid = centeroidpython(data["people"][0]["hand_right_keypoints_2d"])
                            right_hand_window_centroids.append(right_hand_centroid)
                        #print data["people"][0]["hand_left_keypoints_2d"]
                        if centeroidpython(data["people"][0]["hand_left_keypoints_2d"]).count(centeroidpython(data["people"][0]["hand_left_keypoints_2d"])[0]) != len(centeroidpython(data["people"][0]["hand_left_keypoints_2d"])) and centeroidpython(data["people"][0]["hand_left_keypoints_2d"])[0] != 0:
                            left_hand_centroid = centeroidpython(data["people"][0]["hand_left_keypoints_2d"])
                            left_hand_window_centroids.append(left_hand_centroid)

                        #print right_hand_centroid, left_hand_centroid
                        #right_hand_window_centroids.append(right_hand_centroid)
                        #left_hand_window_centroids.append(left_hand_centroid)

                not_enough_datapoints = False
                try:
                    if len(right_hand_window_centroids) > 2:
                        right_hand_bounding_box = MinimumBoundingBox(tuple(right_hand_window_centroids))
                    else:
                        not_enough_datapoints = True
                    #print "centroids", left_hand_window_centroids
                    if len(left_hand_window_centroids) > 2:
                        left_hand_bounding_box = MinimumBoundingBox(tuple(left_hand_window_centroids))
                    else:
                        not_enough_datapoints = True
                except:
                    print "something happened during min bound box claculation"
                    not_enough_datapoints = True
            
                if not not_enough_datapoints:
                    right_hand_distance = max(right_hand_bounding_box.length_parallel, right_hand_bounding_box.length_orthogonal)
                    left_hand_distance = max(left_hand_bounding_box.length_parallel, right_hand_bounding_box.length_orthogonal)
                    hands_distance_max = max(right_hand_distance, left_hand_distance)

                    if last_point != None:
                        find_slope = linregress([last_point[0], int(a_file.split("-")[1].split("_")[0])], [last_point[1], int(hands_distance_max)])
                        #print "slope", find_slope.slope, "for frame", a_file.split("-")[1].split("_")[0]
                    #print "distance", hands_distance_max, "for file", a_file
                    both_hands_distances_traveled_per_sign.append((str(int(a_file.split("-")[1].split("_")[0])),int(hands_distance_max)))
                    last_point = (int(a_file.split("-")[1].split("_")[0]),int(hands_distance_max))


        #if graph_index % graphs_per_page == 0:
            #print "ax available"
            #fig, ax = plt.subplots(int(math.sqrt(graphs_per_page)), int(math.sqrt(graphs_per_page)), figsize=(15,15))

        if True:#else:
            #print "setting a graph"
            #print "before"
            distances_bucket_counter = [ [0,[]] for x in range( total_buckets ) ] #[[0,[]]]*10#,[0,[]],[0,[]],[0,[]],[0,[]],[0,[]],[0,[]],[0,[]],[0,[]],[0,[]]]
            for file_distance in both_hands_distances_traveled_per_sign:
                counter_position = file_distance[1] / bucket_size
                if counter_position > (total_buckets-1): counter_position=total_buckets-1
                distances_bucket_counter[counter_position][0] += 1
                distances_bucket_counter[counter_position][1].append(str(file_distance[0])+"-"+str(int(file_distance[0])+window_size)
)

            #labels, values = zip(*Counter([i[1] for i in both_hands_distances_traveled_per_sign]).items())

            indexes = np.arange(len(distances_bucket_counter))
            width = 1

            #print "graph_index", graph_index, "add to", graph_index / int(math.sqrt(graphs_per_page)), graph_index % int(math.sqrt(graphs_per_page))
            #ax[graph_index / int(math.sqrt(graphs_per_page)), graph_index % int(math.sqrt(graphs_per_page))].plot([i[0] for i in both_hands_distances_traveled_per_sign], [i[1]/bucket_size if i[1]/bucket_size < total_buckets-1 else total_buckets-1 for i in both_hands_distances_traveled_per_sign])#bar(indexes+0.35, [i[0] for i in distances_bucket_counter], 0.35)
            #ax[graph_index / int(math.sqrt(graphs_per_page)), graph_index % int(math.sqrt(graphs_per_page))].set_xticks(indexes + width * 0.5)
            #x_tick_labels=[str(x*bucket_size)+"-"+str(x*bucket_size+bucket_size) for x in range(total_buckets)]
            #ax[graph_index / int(math.sqrt(graphs_per_page)), graph_index % int(math.sqrt(graphs_per_page))].set_xticklabels(x_tick_labels, rotation='vertical')
            #ax[graph_index / int(math.sqrt(graphs_per_page)), graph_index % int(math.sqrt(graphs_per_page))].set_title(a_folder.split("/")[-1], fontdict={'fontsize': 6})

            slope_up_search = [(i[0],i[1]/bucket_size) if i[1]/bucket_size < total_buckets-1 else (i[0],total_buckets-1) for i in both_hands_distances_traveled_per_sign]
            slope_down_search = [(i[0],i[1]/bucket_size) if i[1]/bucket_size < total_buckets-1 else (i[0],total_buckets-1) for i in both_hands_distances_traveled_per_sign]

            #print slope_up_search
            if len(slope_up_search) < 3:
                #print "0"
                continue
            slope_up_search = slope_up_search[:int(len(slope_up_search)//2.5)]
            slope_down_search = slope_down_search[int(len(slope_down_search)//1.5):]
            slope_up_max = max(slope_up_search,key=itemgetter(1))
            slope_down_min = max(slope_down_search,key=itemgetter(1))
            #ax[graph_index / int(math.sqrt(graphs_per_page)), graph_index % int(math.sqrt(graphs_per_page))].plot([slope_up_max[0],slope_down_min[0]], [slope_up_max[1],slope_down_min[1]], 'ro')

            #print "cut at frames", int(slope_up_max[0]), "and", int(slope_down_min[0])
            #plt.show()
        if a_folder.split("/")[-1] in cut_off_points:
            print "overriding"
        cut_off_points[a_folder.split("/")[-1]] = (slope_up_max[0], slope_down_min[0])

        #autolabel(rects1, [i[1] for i in distances_bucket_counter], graph_index)

    with open('speeds_barchart_cutoff_points.pickle', 'wb') as handle:
        pickle.dump(cut_off_points, handle, protocol=pickle.HIGHEST_PROTOCOL)


if os.path.isfile("speeds_barchart.pickle") and os.path.isfile("speeds_barchart_cutoff_points.pickle") and exec_option==3:
    resulting_ds = {}

    with open("speeds_barchart.pickle", "rb") as input_file:
        dir_files_dict = pickle.load(input_file)

    with open("speeds_barchart_cutoff_points.pickle", "rb") as input_file:
        cut_off_points = pickle.load(input_file)

    for folder_idx, a_folder in enumerate(dir_files_dict):
        print "folder", a_folder

        resulting_ds[a_folder.split("/")[-1]] = {}
        #if folder_idx > graphs_per_page+1: continue #temporary

        #new_image = np.zeros(shape=(frame_size_wh[1], frame_size_wh[0]))

        frames_splits = [-1,-1]
        if a_folder.split("/")[-1] in cut_off_points:
            if True:#if a_file.split("-")[1].split("_")[0] in cut_off_points[a_folder.split("/")[-1]]:
                frames_splits[0] = cut_off_points[a_folder.split("/")[-1]][0]
                frames_splits[1] = cut_off_points[a_folder.split("/")[-1]][1]

        for idx, a_file in enumerate(dir_files_dict[a_folder]):
            if not os.path.exists('new_ds/'+a_folder.split("/")[-1]+"/left/"):
                os.makedirs('new_ds/'+a_folder.split("/")[-1]+"/left/")
            if not os.path.exists('new_ds/'+a_folder.split("/")[-1]+"/right/"):
                os.makedirs('new_ds/'+a_folder.split("/")[-1]+"/right/")


            resulting_ds[a_folder.split("/")[-1]][a_file.split("-")[1].split("_")[0]] = [None, None]

            new_image_right = np.zeros(shape=(frame_size_wh[1], frame_size_wh[0]))
            new_image_left = np.zeros(shape=(frame_size_wh[1], frame_size_wh[0]))

            #print "landed on file", a_file
            if True:#int(dir_files_dict[a_folder][-1].split("-")[1].split("_")[0]) >= int(a_file.split("-")[1].split("_")[0])+window_size:
                if int(a_file.split("-")[1].split("_")[0]) > int(frames_splits[0]) and int(a_file.split("-")[1].split("_")[0]) < int(
frames_splits[1]): #True:#for window_idx in range(0, window_size):
                    #print "file", a_file
                    with open(a_folder+"/"+a_file) as f:
                        data = json.load(f)


                        k=3
                        del data["people"][0]["hand_right_keypoints_2d"][k-1::k]
                        del data["people"][0]["hand_left_keypoints_2d"][k-1::k]

                        right_hand_chunks = list(chunks(data["people"][0]["hand_right_keypoints_2d"], 2))
                        left_hand_chunks = list(chunks(data["people"][0]["hand_left_keypoints_2d"], 2))
                        #print "right", right_hand_chunks
                        #print "left", left_hand_chunks

                        hand_outside_the_frame_boundaries = False
                        for item in right_hand_chunks:
                            if item[0] > frame_size_wh[0] or item[0] < 0 or item[1] > frame_size_wh[1] or item[1] < 0:
                                hand_outside_the_frame_boundaries = True
                        for item in left_hand_chunks:
                            if item[0] > frame_size_wh[0] or item[0] < 0 or item[1] > frame_size_wh[1] or item[1] < 0: 
                                hand_outside_the_frame_boundaries = True

                        if not hand_outside_the_frame_boundaries:
                            #print "file to save into the new DS", a_file

                            for right_hand_idx, right_hand_point in enumerate(right_hand_chunks):
                                for connections in openpose_keypoints_connections:
                                    if right_hand_idx == connections[0]:
                                        #print right_hand_idx, connections[1], "are connected"
                                        #print [int(right_hand_point[0]), int(right_hand_chunks[connections[1]][0])], [int(right_hand_point[1]), int(right_hand_chunks[connections[1]][1])]
                                        if int(right_hand_point[0]) != 0 and int(right_hand_chunks[connections[1]][0]) != 0 and int(right_hand_point[1]) != 0 and int(right_hand_chunks[connections[1]][1]) != 0:
                                            line_points = get_line_between_2_points([int(right_hand_point[0]), int(right_hand_chunks[connections[1]][0])], [int(right_hand_point[1]), int(right_hand_chunks[connections[1]][1])], 20)
                                            #print "line points", line_points
                                            for a_coordinate in line_points:
                                                if len(new_image_right) > int(a_coordinate[1]) and len(new_image_right[0]) > int(a_coordinate[0]):
                                                    new_image_right[int(a_coordinate[1])][int(a_coordinate[0])] = 1


                            for left_hand_idx, left_hand_point in enumerate(left_hand_chunks):
                                for connections in openpose_keypoints_connections:
                                    if left_hand_idx == connections[0]:
                                        #print right_hand_idx, connections[1], "are connected"
                                        #print [int(left_hand_point[0]), int(left_hand_chunks[connections[1]][0])], [int(left_hand_point[1]), int(left_hand_chunks[connections[1]][1])]
                                        if int(left_hand_point[0]) != 0 and int(left_hand_chunks[connections[1]][0]) != 0 and int(left_hand_point[1]) != 0 and int(left_hand_chunks[connections[1]][1]) != 0:
                                            line_points = get_line_between_2_points([int(left_hand_point[0]), int(left_hand_chunks[connections[1]][0])], [int(left_hand_point[1]), int(left_hand_chunks[connections[1]][1])], 20)
                                            #print "line points", line_points
                                            #print "line between", [int(left_hand_point[0]), int(left_hand_chunks[connections[1]][0])], [int(left_hand_point[1]), int(left_hand_chunks[connections[1]][1])], "is", line_points
                                            for a_coordinate in line_points:
                                                if len(new_image_left) > int(a_coordinate[1]) and len(new_image_left[0]) > int(a_coordinate[0]):
                                                    new_image_left[int(a_coordinate[1])][int(a_coordinate[0])] = 1



                            #if np.count_nonzero(new_image_right) > 0:
                            #    resulting_ds[a_folder.split("/")[-1]][a_file.split("-")[1].split("_")[0]][0] = new_image_right
                            #if np.count_nonzero(new_image_left) > 0:
                            #    resulting_ds[a_folder.split("/")[-1]][a_file.split("-")[1].split("_")[0]][1] = new_image_left
                            if np.count_nonzero(new_image_right) > 0:
                                most_likely_window = [-1, None]
                                for (x, y, window) in sliding_window(new_image_right, stepSize=32, windowSize=(128, 128)):
                                    if np.count_nonzero(window) > most_likely_window[0]:
                                        most_likely_window[0] = np.count_nonzero(window)
                                        most_likely_window[1] = window
                                #resulting_ds[a_folder.split("/")[-1]][a_file.split("-")[1].split("_")[0]][1] = most_likely_window[1]
                                scipy.misc.imsave('new_ds/'+a_folder.split("/")[-1]+"/right/"+a_file.split("-")[1].split("_")[0]+'.png', most_likely_window[1])

                            if np.count_nonzero(new_image_left) > 0:
                                most_likely_window = [-1, None]
                                for (x, y, window) in sliding_window(new_image_left, stepSize=32, windowSize=(128,128)):
                                    if np.count_nonzero(window) > most_likely_window[0]:
                                        most_likely_window[0] = np.count_nonzero(window)
                                        most_likely_window[1] = window
                                scipy.misc.imsave('new_ds/'+a_folder.split("/")[-1]+"/left/"+a_file.split("-")[1].split("_")[0]+'.png', most_likely_window[1])

if exec_option==4:
    e = xml.etree.ElementTree.parse('tegnsprag/DTS_phonology.xml').getroot()
    '''for an_entry in e.findall("Entry"):
        print an_entry.findall("SignVideo")[0].text
        if len(an_entry.findall("Phonology")[0].findall("Seq")) == 1:
            for a_seq in an_entry.findall("Phonology")[0].findall("Seq"):
                if len(a_seq.findall("SignType")) == 1:
                    if True:#if "".join(map(itemgetter(0), groupby(list(a_seq.findall("SignType")[0].text.encode('latin-1'))))).rstri
p(string.digits) in both_hands_signtype:
                        all_sign_types.append(a_seq.findall("SignType")[0].text.encode("latin-1"))
    print set(all_sign_types)
    asd'''

    for (dirpath, dirnames, filenames) in os.walk("/home/bmocialov/new_ds"):
        list_of_files = []
        for filename in filenames:
            if filename.endswith('.png'):
                #print dirpath, filename

                for an_entry in e.findall("Entry"):
                    #print an_entry.findall("SignVideo")[0].text
                    #print an_entry.findall("SignVideo")[0].text.split(".")[0], "==", dirpath.split("/")[-2]
                    if an_entry.findall("SignVideo")[0].text != None and an_entry.findall("SignVideo")[0].text.split(".")[0] == dirpath.split("/")[-2]:
                        if len(an_entry.findall("Phonology")[0].findall("Seq")) == 1:
                            for a_seq in an_entry.findall("Phonology")[0].findall("Seq"):
                                if not os.path.exists("new_ds_classes/"+a_seq.findall("Handshape1")[0].text.encode("latin-1")):
                                    os.makedirs("new_ds_classes/"+a_seq.findall("Handshape1")[0].text.encode("latin-1"))

                                if len(a_seq.findall("SignType")) == 1:
                                    if "".join(map(itemgetter(0), groupby(list(a_seq.findall("SignType")[0].text.encode('latin-1'))))).rstrip(string.digits) in both_hands_signtype:
                                        #both hands for this video
                                        #print "both hands", a_seq.findall("Handshape1")[0].text.encode("latin-1")
                                        copyfile(dirpath+"/"+filename, "new_ds_classes/"+a_seq.findall("Handshape1")[0].text.encode("latin-1")+"/"+dirpath.split("/")[-2]+"_"+filename.split(".")[0]+"_"+dirpath.split("/")[-1]+".png")
                                    elif a_seq.findall("SignType")[0].text.encode('latin-1') in one_hand_signtype:
                                        if len(a_seq.findall("HandshapeFinal")) > 0:
                                            if a_seq.findall("Handshape1")[0].text.encode("latin-1") == a_seq.findall("HandshapeFinal")[0].text.encode("latin-1"):
                                                #both hands for this video
                                                #print "dominant hand", a_seq.findall("Handshape1")[0].text.encode("latin-1")
                                                if "right" in dirpath.split("/"):
                                                    #print "copy", filename, "from", dirpath
                                                    copyfile(dirpath+"/"+filename, "new_ds_classes/"+a_seq.findall("Handshape1")[0].text.encode("latin-1")+"/"+dirpath.split("/")[-2]+"_"+filename.split(".")[0]+"_"+dirpath.split("/")[-1]+".png")
                                        else:
                                            #print "dominant hand", a_seq.findall("Handshape1")[0].text.encode("latin-1")
                                            if "right" in dirpath.split("/"):
                                                #print "copy", filename, "from", dirpath
                                                copyfile(dirpath+"/"+filename, "new_ds_classes/"+a_seq.findall("Handshape1")[0].text.encode("latin-1")+"/"+dirpath.split("/")[-2]+"_"+filename.split(".")[0]+"_"+dirpath.split("/")[-1]+".png")


if exec_option==5:
    old_ds = {}
    new_ds_overall = {}
    new_ds_training = {}
    new_ds_validation = {}
    new_ds_testing = {}
    for (dirpath, dirnames, filenames) in os.walk("/home/bmocialov/new_ds_classes/"):
        list_of_files = []
        old_ds[dirpath.split("/")[-1]] = []
        for filename in filenames:
            if filename.endswith('.png'):
                #print "dir, file", dirpath, filename
                old_ds[dirpath.split("/")[-1]].append(dirpath+"/"+filename)
        #print ("old ds class", dirpath.split("/")[-1], "size", len(old_ds[dirpath.split("/")[-1]]))

    for a_class in old_ds:
        shuffle(old_ds[a_class])
        new_ds_overall[a_class] = old_ds[a_class]
        #print ("overall shuffled ds class", a_class, "size", len(new_ds_overall[a_class]))
    
    for a_class in new_ds_overall:
        if a_class != '':
            new_ds_training[a_class] = new_ds_overall[a_class][:int(int(len(new_ds_overall[a_class])) * .67)]
            new_ds_testing[a_class] = new_ds_overall[a_class][int(int(len(new_ds_overall[a_class])) * .67):]
            #print ("training ds class", a_class, "size", len(new_ds_training[a_class]))

    for a_class in new_ds_testing:
        if a_class != '':
            new_ds_validation[a_class] = new_ds_testing[a_class][:int(int(len(new_ds_testing[a_class])) * .5)]
            new_ds_testing[a_class] = new_ds_testing[a_class][int(int(len(new_ds_testing[a_class])) * .5):]
            #print ("validation ds class", a_class, "size", len(new_ds_validation[a_class]))
            #print ("testing ds class", a_class, "size", len(new_ds_testing[a_class]))

    #saving
    for a_class in new_ds_training:
        if not os.path.exists("new_ds_classes_split/training/"+a_class):
            os.makedirs("new_ds_classes_split/training/"+a_class)
        for a_file in new_ds_training[a_class]:
            #print "from", a_file, "to", "new_ds_classes_split/training/"+a_class+"/"+a_file.split("/")[-1]
            copyfile(a_file, "new_ds_classes_split/training/"+a_class+"/"+a_file.split("/")[-1])

    for a_class in new_ds_validation:
        if not os.path.exists("new_ds_classes_split/validation/"+a_class):
            os.makedirs("new_ds_classes_split/validation/"+a_class)
        for a_file in new_ds_validation[a_class]:
            copyfile(a_file, "new_ds_classes_split/validation/"+a_class+"/"+a_file.split("/")[-1])
    
    for a_class in new_ds_testing:
        if not os.path.exists("new_ds_classes_split/testing/"+a_class):
            os.makedirs("new_ds_classes_split/testing/"+a_class)
        for a_file in new_ds_testing[a_class]:
            copyfile(a_file, "new_ds_classes_split/testing/"+a_class+"/"+a_file.split("/")[-1])

json_dataset_filename = "json_dataset"
if os.path.isfile("speeds_barchart.pickle") and exec_option==6:
    e = xml.etree.ElementTree.parse('tegnsprag/DTS_phonology.xml').getroot()

    with open("speeds_barchart.pickle", "rb") as input_file:
        dir_files_dict = pickle.load(input_file)

    for folder_idx, a_folder in enumerate(dir_files_dict):
        #print a_folder, dir_files_dict[a_folder]  #json file names mapping[folder]=files_frame#

        for (dirpath, dirnames, filenames) in os.walk("/home/bmocialov/new_ds_classes_split"):
            list_of_files = []
            for filename in filenames:
                if filename.endswith('.png'):
                    #print a_folder +"=="+ filename
                    #/filenameadas
                    if a_folder.split("/")[-1] == filename.split("_")[0]+"_"+filename.split("_")[1]:
                        #print filename, a_folder
                        #asd
                        #with open(json_dataset_filename+"_"+dirpath.split("_")[-2]+".txt", 'a') as file:
                        #    file.write(" "+dirpath.split("_")[-1]+"\n")
                        for an_entry in e.findall("Entry"):
                            if an_entry.findall("SignVideo")[0].text != None and an_entry.findall("SignVideo")[0].text.split(".")[0] == filename.split("_")[0]+"_"+filename.split("_")[1]:
                                if len(an_entry.findall("Phonology")[0].findall("Seq")) == 1:
                                    for a_seq in an_entry.findall("Phonology")[0].findall("Seq"):
                                        if len(a_seq.findall("SignType")) == 1:
                                            if "".join(map(itemgetter(0), groupby(list(a_seq.findall("SignType")[0].text.encode('latin-1'))))).rstrip(string.digits) in both_hands_signtype:
                                                #both hands for this video
                                                #print "both hands for", filename, a_folder
                                                for a_json_file in dir_files_dict[a_folder]:
                                                    if filename.split("_")[2] == a_json_file.split("-")[1].split("_")[0]:
                                                        with open(a_folder + "/" + a_json_file) as afile:
                                                            #print "both hands for", filename, a_folder
                                                            data = json.load(afile)
                                                            #print data["people"][0]["hand_right_keypoints_2d"]
                                                            #print data["people"][0]["hand_left_keypoints_2d"]
                                                            if not csv_file_contains(json_dataset_filename+"_"+dirpath.split("/")[-2]+".txt", a_folder + "/" + a_json_file):
                                                                with open(json_dataset_filename+"_"+dirpath.split("/")[-2]+".txt", 'a') as aafile:
                                                                    aafile.write(a_folder+"/"+a_json_file+","+",".join(map(str, data["people"][0]["hand_right_keypoints_2d"]))+","+dirpath.split("/")[-1]+"\n")
                                                                    aafile.write(a_folder+"/"+a_json_file+","+",".join(map(str, data["people"][0]["hand_left_keypoints_2d"]))+","+dirpath.split("/")[-1]+"\n")

                                            elif a_seq.findall("SignType")[0].text.encode('latin-1') in one_hand_signtype:
                                                if len(a_seq.findall("HandshapeFinal")) > 0:
                                                    if a_seq.findall("Handshape1")[0].text.encode("latin-1") == a_seq.findall("HandshapeFinal")[0].text.encode("latin-1"):
                                                        #FinalHandshape is present and it is the same as the starting handshape
                                                        for a_json_file in dir_files_dict[a_folder]:
                                                            if filename.split("_")[2] == a_json_file.split("-")[1].split("_")[0]:
                                                                with open(a_folder + "/" + a_json_file) as afile:
                                                                    #print "dominant hands for", filename, a_folder
                                                                    data = json.load(afile)
                                                                    #print data["people"][0]["hand_right_keypoints_2d"]
                                                                    if not csv_file_contains(json_dataset_filename+"_"+dirpath.split("/")[-2]+".txt", a_folder + "/" + a_json_file):
                                                                        with open(json_dataset_filename+"_"+dirpath.split("/")[-2]+".txt", 'a') as aafile:
                                                                            aafile.write(a_folder+"/"+a_json_file+","+",".join(map(str, data["people"][0]["hand_right_keypoints_2d"]))+","+dirpath.split("/")[-1]+"\n")



                                                else:
                                                    #final handshape is the same as the sarting handshape (no final handshape)
                                                    for a_json_file in dir_files_dict[a_folder]:
                                                        if filename.split("_")[2] == a_json_file.split("-")[1].split("_")[0]:
                                                            with open(a_folder + "/" + a_json_file) as afile:
                                                                #print "dominant hands for", filename, a_folder
                                                                data = json.load(afile)
                                                                #print data["people"][0]["hand_right_keypoints_2d"]
                                                                if not csv_file_contains(json_dataset_filename+"_"+dirpath.split("/")[-2]+".txt", a_folder + "/" + a_json_file):
                                                                    with open(json_dataset_filename+"_"+dirpath.split("/")[-2]+".txt", 'a') as aafile:
                                                                        aafile.write(a_folder+"/"+a_json_file+","+",".join(map(str, data["people"][0]["hand_right_keypoints_2d"]))+","+dirpath.split("/")[-1]+"\n")


if os.path.isfile("speeds_barchart.pickle") and os.path.isfile("speeds_barchart_cutoff_points.pickle") and exec_option==7:
    resulting_ds = {}

    with open("speeds_barchart.pickle", "rb") as input_file:
        dir_files_dict = pickle.load(input_file)

    with open("speeds_barchart_cutoff_points.pickle", "rb") as input_file:
        cut_off_points = pickle.load(input_file)

