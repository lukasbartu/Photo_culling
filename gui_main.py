__author__ = 'Lukáš Bartůněk'

import PySimpleGUI as sg
from quality_assessment import prepare_model, calculate_qualities
from similarity_assessment import calculate_similarities
from content_assessment import calculate_content
from summary_creation import select_summary
from utils import prepare_paths,prepare_img_list
import time

layout = [
    [
        sg.Push(),
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
        sg.Push()
    ],[
        sg.Text("Number of neighbours"),
        sg.Push(),
        sg.Slider((0,50), orientation='h', s=(10,15),default_value=10, resolution=5, tooltip="Recommended: 10",key="-NBRS"),
    ],[
        sg.Text("Similarity threshold"),
        sg.Push(),
        sg.Slider((0, 50), orientation='h', s=(10, 15), default_value=10,resolution=10, tooltip="Recommended: 10",key="-S_T"),
    ],[
        sg.Text("Technical quality weight of summary selection"),
        sg.Push(),
        sg.Slider((0,100), orientation='h', s=(10,15),default_value=50,resolution=5, tooltip="Recommended: 50%",key="-T_A_RATIO")
    ],[
        sg.Text("Importance of image content"),
        sg.Push(),
        sg.Slider((0, 100), orientation='h', s=(10, 15), default_value=50,resolution=5, tooltip="Recommended: 50%", key="-C_Q_RATIO"),
    ],[
        sg.Text("Percentage of all images in selection"),
        sg.Push(),
        sg.Slider((0,100), orientation='h', s=(10,15),default_value=10,resolution=5, tooltip="Recommended: 10%",key="-PERCENT"),
    ],[
        sg.Push(),
        sg.Button("Generate summary",key="-SUMM"),
        sg.Push()
    ],[
        [sg.Output(size=(60,4),key="-OUTPUT")],
    ]
]

# Create the window
window = sg.Window("Photo selector", layout)

folder = []
img_list = []
sim_path = None
q_path = None
c_path = None
img_num = None
percent = None
c_q_ratio = None
# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window
    if event == sg.WIN_CLOSED:break
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        _, sim_path, q_path, c_path = prepare_paths(folder,abs_p=True)
        img_list, img_num = prepare_img_list(folder)
    if event == "-SUMM":
        if not folder:
            sg.popup("No folder selected")
        else:
            window.FindElement("-OUTPUT").Update('')
            tic = time.perf_counter()
            print("Calculating quality...", end="   ")
            window.Refresh() if window else None
            calculate_qualities(pth=folder, lst=img_list, result_pth=q_path, model_pth= "model.pth")
            print("Quality calculated")
            window.Refresh() if window else None

            print("Creating content description...", end="   ")
            window.Refresh() if window else None
            calculate_content(pth=folder, lst=img_list, result_pth=c_path)
            print("Content description created")
            window.Refresh() if window else None

            nbrs = values["-NBRS"]
            window.Refresh() if window else None
            print("Calculating similarities...", end="   ")
            window.Refresh() if window else None
            calculate_similarities(pth=folder, lst=img_list, result_pth=sim_path, num=img_num,
                                   nbrs=nbrs,content_pth=c_path)
            print("Similarities calculated")
            window.Refresh() if window else None

            percent = values["-PERCENT"]
            s_t = values["-S_T"]
            c_q_ratio = values["-C_Q_RATIO"]
            t_a_ratio = values["-T_A_RATIO"]
            print("Selecting summary of photos...", end="   ")
            window.Refresh() if window else None
            summary = select_summary(sim_pth=sim_path, q_pth=q_path, c_pth=c_path, percent=percent, num=img_num,
                                     s_t=s_t,dir_pth=folder, c_q_r=c_q_ratio, t_a_r=t_a_ratio)
            print("Summary calculated")
            window.Refresh() if window else None
            toc = time.perf_counter()
            print(f"Process took: {toc - tic:0.2f} s")

window.close()