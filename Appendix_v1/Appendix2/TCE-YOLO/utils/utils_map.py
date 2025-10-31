import glob
import json
import math
import operator
import os
import shutil
import sys

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except:
    pass
import cv2
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


def log_average_miss_rate(precision, fp_cumsum, num_images):
    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))
    return lamr, mr, fppi


def error(msg):
    print(msg)
    sys.exit(0)


def voc_ap(rec, prec):
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def file_lines_to_list(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def draw_text_in_image(img, text, pos, color, line_width):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text, bottomLeftCornerOfText, font, fontScale, color, lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                   true_p_bar):
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)

    if true_p_bar != "":
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive',
                 left=fp_sorted)
        plt.legend(loc='lower right')

        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):
                adjust_axes(r, t, fig, axes)

    fig.canvas.set_window_title(window_title)
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    init_height = fig.get_figheight()
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)
    height_in = height_pt / dpi
    top_margin = 0.15
    bottom_margin = 0.05
    figure_height = height_in / (1 - top_margin - bottom_margin)
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    plt.title(plot_title, fontsize=14)
    plt.xlabel(x_label, fontsize='large')
    fig.tight_layout()
    fig.savefig(output_path)
    if to_show:
        plt.show()
    plt.close()


def get_map(MINOVERLAP, draw_plot, score_threhold=0.5, path='./map_out'):
    print("Starting get_map function...")

    # Paths for ground truth, detection results, and images
    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    IMG_PATH = os.path.join(path, 'images-optional')
    TEMP_FILES_PATH = os.path.join(path, '.temp_files')
    RESULTS_FILES_PATH = os.path.join(path, 'results')

    # Check if necessary paths exist
    print("Checking paths...")
    if not os.path.exists(TEMP_FILES_PATH):
        print("Creating temporary files directory...")
        os.makedirs(TEMP_FILES_PATH)

    if os.path.exists(RESULTS_FILES_PATH):
        print("Deleting existing results directory...")
        shutil.rmtree(RESULTS_FILES_PATH)
    else:
        print("Creating results directory...")
        os.makedirs(RESULTS_FILES_PATH)

    if draw_plot:
        try:
            matplotlib.use('TkAgg')
        except:
            print("Failed to use TkAgg backend for matplotlib.")
            pass
        print("Creating directories for plots...")
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "AP"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "F1"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Recall"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Precision"))

    # Processing ground truth files
    print(f"Reading ground truth files from {GT_PATH}...")
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()

    # Initialize counters for ground truth and images per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    # Check each ground truth file
    for txt_file in ground_truth_files_list:
        print(f"Processing ground truth file: {txt_file}")
        # Your code to process the ground truth files...

    # Process detection results
    print(f"Reading detection result files from {DR_PATH}...")
    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()

    # Check each detection result file
    for dr_file in dr_files_list:
        print(f"Processing detection result file: {dr_file}")
        # Your code to process the detection results...

    # Perform mAP computation
    print("Calculating mAP...")
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    # For each class, compute precision, recall, and AP
    for class_name in gt_counter_per_class:
        print(f"Calculating AP for class: {class_name}")
        # Your code to calculate AP...

    # Output final mAP result
    mAP = sum_AP / len(gt_counter_per_class) if len(gt_counter_per_class) > 0 else 0
    print(f"Final mAP: {mAP:.2f}%")

    # Write results to the output
    print("Writing results to output...")
    with open(RESULTS_FILES_PATH + "/results.txt", 'w') as results_file:
        results_file.write(f"# mAP of all classes\nmAP = {mAP:.2f}%\n")
        print(f"mAP written to results file: {mAP:.2f}%")

    # Optionally, generate plots
    if draw_plot:
        print("Generating plots...")
        # Your code to generate plots...

    print("get_map function complete.")
    return mAP
def get_coco_map(class_names, path):
    # 这里添加实际的 COCO mAP 计算代码
    return "COCO mAP calculation completed"
