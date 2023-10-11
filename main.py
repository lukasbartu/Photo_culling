__author__ = 'Lukáš Bartůněk'

# Select a subset of photographs from current directory to represent the sequence.
# Considers both technical and aesthetic quality while trying to eliminate duplicity.
#
# SIFT similarity is edited code taken from https://github.com/adumrewal/SIFTImageSimilarity/blob/master/SIFTSimilarityInteractive.ipynb
# NIMA image quality is edited code taken from https://github.com/yunxiaoshi/Neural-IMage-Assessment
# argument parser code taken from https://gitlab.fel.cvut.cz/kybicjan/fotoselektor
#
# DONE - Familiarize yourself with existing tools and approaches in this domain
#           and suggest a suitable architecture for the photo selection tool.
# DONE - Test existing approaches for evaluating image quality, image similarity and  image content.
# DONE - Evaluate several methods for combining extracted image information for photo selection.
# DONE - Implement an easy-to-use software tool for photo  selection, allowing the user to set basic parameters of the process.
# TODO - Experimentally evaluate the performance of the developed software by comparing it with other existing tools,
#           by comparing it with human performance, or by evaluating user satisfaction.
# TODO - [Optional] Add the ability to read and write image metadata for easier integration with standard tools.
# TODO - [Optional] Create a graphical user interface allowing to interactively examine and alter the selection process.
# TODO - [Optional] Improve the selection process by learning from a (possibly large) dataset of manually selected photographs.


import time
import argparse
from quality_assessment import prepare_model, calculate_qualities
from similarity_assessment import calculate_similarities
from content_assessment import calculate_content
from summary_creation import select_summary
from utils import prepare_paths,prepare_img_list

def main(arg_list):
    abs_pth,sim_path,q_path,c_path = prepare_paths(arg_list.directory,abs_p=False)
    img_list,img_num = prepare_img_list(abs_pth)
    c_q_ratio = arg_list.content_weight/100
    t_a_ratio = arg_list.technical_weight/100
    percent = arg_list.percentage
    s_t = arg_list.similarity_threshold
    nbrs = arg_list.number_of_neighbours
    model_path = arg_list.model_path

    if arg_list.calculate_quality:
        tic = time.perf_counter()
        calculate_qualities(pth=abs_pth, lst=img_list, result_pth=q_path, model_pth=model_path)
        print("Quality Calculated")
        toc = time.perf_counter()
        print(f"Process took: {toc - tic:0.2f} s")
    if arg_list.calculate_similarity:
        tic = time.perf_counter()
        print("Images have been loaded")
        calculate_similarities(pth=abs_pth, lst=img_list, result_pth=sim_path, num=img_num, nbrs=nbrs)
        print("Similarities calculated")
        toc = time.perf_counter()
        print(f"Process took: {toc - tic:0.2f} s")
    if arg_list.calculate_content:
        print("Calculate content")
        calculate_content(abs_pth, img_list, c_path)
        print("Content Calculated")
    if arg_list.select_photos:
        tic = time.perf_counter()
        print("Selecting summary of photos")
        summary = select_summary(sim_pth=sim_path, q_pth=q_path, c_pth=c_path, percent=percent, num=img_num, s_t=s_t,
                                 dir_pth=abs_pth, c_q_r=c_q_ratio)
        print("Summary:", summary)
        toc = time.perf_counter()
        print(f"Process took: {toc - tic:0.2f} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a subset of photographs from current directory. Considers both technical and aesthetic quality as well as similarity between photographs.",
                                     epilog='Lukáš Bartůněk, 2023')
    parser.add_argument('-q','--calculate_quality',
                        help='precalculate aesthetic and technical quality',action='store_true')
    parser.add_argument('-s','--calculate_similarity',
                        help='precalculate similarity between images',action='store_true')
    parser.add_argument('-c', '--calculate_content',
                        help='precalculate image contents', action='store_true')
    parser.add_argument('-e','--select_photos',
                        help='select a subset of images based on the precalculated quality and similarities',action='store_true')
    parser.add_argument('-n','--number_of_neighbours',
                        help='how many images before and after to consider for similarity calculation',type=int,default=5)
    parser.add_argument('-t_weight','--technical_weight',
                        help='how much weight (0-100) to give to the technical quality. The remainder is aesthetic quality',
                        type=int,default=50)
    parser.add_argument('-c_weight', '--content_weight',
                        help='how much weight (0-100) to give to the quality. The remainder is user preference score',
                        type=int, default=50)
    parser.add_argument('-p','--percentage',
                        help='how many percent (0-100) of the original images to select in the first round.',type=int,default=10)
    parser.add_argument('-s_t','--similarity_threshold',
                        help='threshold on SIFT similarity to consider images as similar to be pruned.',type=int,default=10)
    parser.add_argument('-model','--model_path',
                        help='path to model for quality assessment',default= "model.pth" )
    parser.add_argument( '-dir','--directory',
                         help='directory containing photographs',default= "/images/Ples/fotokoutek" )
    args=parser.parse_args()
    main(args)