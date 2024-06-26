__author__ = 'Lukáš Bartůněk'

# Select a subset of photographs from current directory to represent the sequence.
# Considers both technical and aesthetic quality while trying to eliminate duplicity.
#
# SIFT similarity is edited code taken from:
# https://github.com/adumrewal/SIFTImageSimilarity/blob/master/SIFTSimilarityInteractive.ipynb

# NIMA image quality is edited code taken from:
# https://github.com/yunxiaoshi/Neural-IMage-Assessment

from quality_assessment import calculate_qualities
from similarity_assessment import calculate_similarities
from content_assessment import calculate_content
from summary_creation import select_summary
from utils import prepare_paths, prepare_img_list, remove_folder_name, copy_images, save_list, load_trained
from metadata_creation import include_metadata_rating
import logical_approximation
import neural_network

import time
import argparse
import os


def main(arg_list):
    q_t, s_t, t_a_ratio, s_c_ratio, size = None, None, None, None, None
    print(" ***This is a console version of this software. This version lacks some features that are included in "
          "the GUI version. For full functionality use gui_main.py. ***")
    if not os.path.isdir(arg_list.directory):
        print("Not a directory!")
        return 1
    sim_path, q_path, c_path = prepare_paths(arg_list.directory)
    img_list, img_num = prepare_img_list(arg_list.directory)
    if img_num == 0:
        print("Directory does not contain any images or images are in wrong format. "
              "[Software only supports .jpg and .jpeg]")
        return 1

    size_based = arg_list.size_based_selection
    nbrs = 20

    summary = []
    if (arg_list.select_photos_man or arg_list.select_photos_recommended
            or arg_list.select_photos_log or arg_list.select_photos_nn):
        if arg_list.select_photos_recommended:
            q_t, s_t, t_a_ratio, s_c_ratio = load_trained()
        else:
            q_t = max(min(arg_list.quality_threshold, 100), 0)
            s_t = max(min(arg_list.similarity_threshold, 100), 0)
            t_a_ratio = max(min(arg_list.t_a_ratio, 100), 0) / 100
            s_c_ratio = max(min(arg_list.s_c_ratio, 100), 0) / 100
        if size_based:
            if size is None:
                print("Please enter size [-size] for size based selection.")
                return 1
            else:
                output_size = int(img_num * (size / 100))
        else:
            output_size = 0
            size = 1
        tic = time.perf_counter()
        print("Calculating quality...", end="   ")
        calculate_qualities(lst=img_list, result_pth=q_path)
        print("Quality Calculated")
        print("Calculating content...", end="   ")
        calculate_content(lst=img_list, result_pth=c_path)
        print("Content Calculated")
        print("Calculating similarities...", end="   ")
        calculate_similarities(lst=img_list, result_pth=sim_path, num=img_num, nbrs=nbrs, content_pth=c_path)
        print("Similarities calculated")
        print("Selecting selection of photos")
        if arg_list.select_photos_man or arg_list.select_photos_recommended:
            summary = select_summary(sim_pth=sim_path, q_pth=q_path, size=size, num=img_num, s_t=s_t,
                                     t_a_ratio=t_a_ratio,
                                     size_based=size_based, q_cutoff=q_t, s_c_ratio=s_c_ratio)

        elif arg_list.select_photos_log:
            summary = logical_approximation.summary(lst=img_list, s_file=sim_path, q_file=q_path,
                                                    size_based=size_based, output_size=output_size)
            weights = logical_approximation.load_weights()
            t_a_ratio = weights[0].item()
        elif arg_list.select_photos_nn:
            summary = neural_network.summary(lst=img_list, s_file=sim_path, q_file=q_path, size_based=size_based,
                                             output_size=output_size)
            t_a_ratio = 50

        summary = remove_folder_name(summary, (os.getcwd() + "/" + arg_list.directory))
        print("Selection:", summary)
        toc = time.perf_counter()
        print(f"Process took: {toc - tic:0.2f} s")

        if arg_list.metadata:
            print("Writing metadata...", end="   ")
            include_metadata_rating(img_list=img_list, q_file=q_path, t_a_ratio=t_a_ratio)
            print("Metadata written")

        if arg_list.save_selection_list:
            save_list(summary=summary, folder=arg_list.directory, dest_folder=arg_list.directory)
            print("Selection list saved into:", arg_list.directory)

        if arg_list.copy_selection:
            copy_images(summary=summary, folder=arg_list.directory, dest_folder="Selected images/")
            print("Selection copied into:", "Selected images/")

    else:
        print("Select one of the modes: [-man_cond]/[-auto_log]/[-auto_nn].")
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a subset of photographs from current directory. "
                    "Considers both technical and aesthetic quality as well as similarity between photographs."
                    "To access all the available functions use GUI version of this software.",
        epilog='Lukáš Bartůněk, 2023')
    parser.add_argument('-man_cond', '--select_photos_man',
                        help='Selection based on provided parameters.', action='store_true')

    parser.add_argument('-recommended', '--select_photos_recommended',
                        help='Selection based on recommended parameters.', action='store_true')

    parser.add_argument('-auto_log', '--select_photos_log',
                        help='Selection based on automatic parameters learned by logical approximation.',
                        action='store_true')

    parser.add_argument('-auto_nn', '--select_photos_nn',
                        help='Selection based on automatic parameters learned by neural network.',
                        action='store_true')

    parser.add_argument('-t_a_ratio', '--t_a_ratio',
                        help='How much weight (0-100) to give to technical quality. '
                             'The remainder is aesthetic quality. [0:100]',
                        type=int, default=50)

    parser.add_argument('-s_c_ratio', '--s_c_ratio',
                        help='How much weight (0-100) to give to structural similarity. '
                             'The remainder is content similarity. [0:100]',
                        type=int, default=50)

    parser.add_argument('-size', '--size',
                        help='How many percent (0-100) of the original images to select in the first round. [0:100]',
                        type=int, default=10)

    parser.add_argument('-s_t', '--similarity_threshold',
                        help='Threshold on SIFT similarity to consider images as similar to be pruned. [0:100]',
                        type=int, default=10)

    parser.add_argument('-q_t', '--quality_threshold',
                        help='Threshold of quality that will be included in the selection. [0:100]', type=int, default=50)

    parser.add_argument("-metadata", "--metadata", action="store_true",
                        help="Write information about quality of images into metadata.")

    parser.add_argument('-save', '--save_selection_list',
                        help='Saves the selection as a list in .txt.', action='store_true')

    parser.add_argument('-copy', '--copy_selection',
                        help='Copies selected images into new folder.', action='store_true')

    parser.add_argument('-size_based', '--size_based_selection',
                        help='Select based on output size.', action='store_true')
    parser.set_defaults(size_based_selection=False)

    parser.add_argument('-dir', '--directory',
                        help='Directory containing images.', required=True)
    args = parser.parse_args()
    main(args)
