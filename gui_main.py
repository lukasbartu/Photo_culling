__author__ = 'Lukáš Bartůněk'

import PySimpleGUI as sg
from quality_assessment import calculate_qualities
from similarity_assessment import calculate_similarities
from content_assessment import calculate_content
from summary_creation import select_summary
from utils import prepare_paths, prepare_img_list, remove_folder_name
from training_parameters import load_trained
from auto_summary import auto_select_summary, update_parameters, load_weights
from metadata_creation import include_metadata_rating
import time
import os
import PIL.Image
import io
import base64
import natsort
import pathlib
import shutil


# PysimpleGUI demo
class BtnInfo:
    def __init__(self, state=True):
        self.state = state  # Can have 3 states - True, False, None (disabled)


# PysimpleGUI demo
def convert_to_bytes(file_or_bytes, resize=None):
    """
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    Turns into  PNG format in the process so that can be displayed by tkinter
    :param file_or_bytes: either a string filename or a bytes base64 image object
    :type file_or_bytes:  (Union[str, bytes])
    :param resize:  optional new size
    :type resize: (Tuple[int, int] or None)
    :return: (bytes) a byte-string object
    :rtype: (bytes)
    """
    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height / cur_height, new_width / cur_width)
        img = img.resize((int(cur_width * scale), int(cur_height * scale)), PIL.Image.LANCZOS)
    with io.BytesIO() as bio:
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()


def make_win1():
    layout = [
        [
            sg.Push(),
            sg.Text("Image Folder"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
            sg.Push()
        ], [
            sg.Text("Number of neighbours"),
            sg.Push(),
            sg.Slider((0, 50), orientation='h', s=(10, 15), default_value=10, resolution=5,
                      tooltip="Recommended: 10\nBased on maximum number of duplicates in a row", key="-NBRS"),
            sg.Push(),
            sg.Checkbox("Recalculate", key="-RECALC", enable_events=True)
        ],
        [sg.HSeparator(), ],
        [
            sg.Push(),
            sg.Text("Similarity threshold"),
            sg.Slider((0, 50), orientation='h', s=(10, 15), default_value=10, resolution=2, tooltip="Recommended: 10",
                      key="-S_T"),
            sg.Push()
        ], [
            sg.Push(),
            sg.Text("Aesthetic quality "),
            sg.Slider((0, 100), orientation='h', s=(10, 15), default_value=50, resolution=2, tooltip="Recommended: 50%",
                      key="-T_A_RATIO"),
            sg.Text("Technical quality "),
            sg.Push(),
        ],
        [sg.HSeparator(), ],
        [
            sg.Push(),
            sg.Text("Selection based on:"),
            sg.Push(),
            sg.Button('Output size', size=(10, 2), button_color='white on green', key='-B-'),
            sg.Push()
        ], [
            sg.Push(),
            sg.Text("Output size in percents"),
            sg.Push(),
            sg.Slider((0, 100), orientation='h', s=(10, 15), default_value=10, resolution=2, tooltip="Recommended: 10%",
                      key="-SIZE"),
            sg.Push(),
        ], [
            sg.Push(),
            sg.Text("Quality threshold"),
            sg.Push(),
            sg.Slider((0, 100), orientation='h', s=(10, 15), default_value=50, resolution=1, key="-QUALITY_CUTOFF"),
            sg.Push(),
        ],
        [sg.HSeparator(), ],
        [
            sg.Push(),
            sg.Text("Summary generation"),
            sg.Push(),
        ], [
            sg.Push(),
            sg.Button("Use selected parameters", key="-SUMM_MAN", size=(40, 1)),
            sg.Push(),
        ], [
            sg.Push(),
            sg.Button("Use recommended parameters", key="-SUMM_REC", size=(40, 1)),
            sg.Push()
        ], [
            sg.Push(),
            sg.Button("Automatic summary with trainable parameters", key="-SUMM_AUTO",
                      tooltip="Only selection based on quality", size=(40, 1)),
            sg.Push()
        ], [
            [sg.Output(size=(60, 4), key="-OUTPUT")],
        ]
    ]
    return sg.Window("Photo selector", layout, finalize=True)


def make_win2():
    left_col = [[sg.Text("Selected images"), ],
                [sg.Listbox(values=[], enable_events=True, size=(40, 25), key='-SUM_LIST'),
                 sg.Button("Deselect image", key="-MOVE_DOWN", size=(10, 2)), ],
                [sg.Text("Not selected images")],
                [sg.Listbox(values=[], enable_events=True, size=(40, 25), key='-REST_LIST'),
                 sg.Button("Select image", key="-MOVE_UP", size=(10, 2)), ]]
    images_col = [[sg.Text('You choose from the list:')],
                  [sg.Push(), sg.Text(key='-TOUT-'), sg.Push()],
                  [sg.Image(key='-IMAGE-', size=(900, 760))]]
    layout = [
        [
            sg.Column(left_col, element_justification='c'),
            sg.VSeparator(),
            sg.Column(images_col, element_justification='c')
        ], [
            sg.Push(),
            sg.Button("Update parameters for automatic selection", key="-UPDATE_PARA", size=(20, 2)),
            sg.In(key="-COPY", visible=False, enable_events=True),
            sg.FolderBrowse("Copy selection into folder", size=(15, 2)),
            sg.Button("Exit", key="-EXIT", size=(10, 2)),
        ]
    ]
    return sg.Window("Selection window", layout, finalize=True)


window_default, window_selection = make_win1(), None

folder = []
img_list = []
sim_path = None
q_path = None
c_path = None
img_num = None
size = None
c_q_ratio = None
selection = True
summ_create = False
auto_summ = False
updated = False
# Create an event loop
while True:
    window, event, values = sg.read_all_windows()
    if event == sg.WIN_CLOSED or event == '-EXIT':
        window.close()
        if window == window_selection:  # if closing win 2, mark as closed
            window_selection = None
        elif window == window_default:  # if closing win 1, exit program
            break
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        _, sim_path, q_path, c_path = prepare_paths(folder, abs_p=True)
        img_list, img_num = prepare_img_list(folder)
    if event == '-B-':
        selection = not selection
        window['-B-'].update(text='Output size' if selection else 'Quality threshold',
                             button_color='white on green' if selection else 'white on blue')
    if event == "-SUMM_REC" or event == "-SUMM_AUTO" or event == "-SUMM_MAN":
        if event == "-SUMM_REC":
            size = values["-SIZE"]
            s_t = values["-S_T"]
            t_a_ratio = values["-T_A_RATIO"] / 100
            q_t = values["-QUALITY_CUTOFF"]
        elif event == "SUMM_MAN":
            q_t, s_t, t_a_ratio, size = load_trained()
        else:
            auto_summ = True
        if not folder:
            sg.popup("No folder selected")
        else:
            summ_create = True
            window["-OUTPUT"].Update('')
            tic = time.perf_counter()
            print("Calculating quality...", end="   ")
            window.Refresh() if window else None
            window.perform_long_operation(lambda: calculate_qualities(lst=img_list, result_pth=q_path,
                                                                      model_pth="data/model.pth"), end_key="-QUA_DONE")
    elif event == "-QUA_DONE" and summ_create:
        print("Quality calculated")
        window.Refresh() if window else None
        print("Creating content description...", end="   ")
        window.Refresh() if window else None
        window.perform_long_operation(lambda: calculate_content(lst=img_list, result_pth=c_path)
                                      , end_key="-CON_DONE")

    elif event == "-CON_DONE" and summ_create:
        print("Content description created")
        window.Refresh() if window else None
        nbrs = int(values["-NBRS"])
        recalc = values["-RECALC"]
        window.Refresh() if window else None
        print("Calculating similarities...", end="   ")
        window.Refresh() if window else None

        window.perform_long_operation(lambda: calculate_similarities(lst=img_list, result_pth=sim_path, num=img_num,
                                                                     nbrs=nbrs, content_pth=c_path, recalc=recalc),
                                      end_key="-SIM_DONE")

    elif event == "-SIM_DONE" and summ_create:
        print("Similarities calculated")
        window.Refresh() if window else None
        print("Selecting summary of photos...", end="   ")
        window.Refresh() if window else None
        if auto_summ:
            summary = auto_select_summary(img_list=img_list, s_file=sim_path, q_file=q_path, nbrs=nbrs)
            weights = load_weights()
            t_a_ratio = weights[0].item()
        else:
            summary = select_summary(sim_pth=sim_path, q_pth=q_path, size= size, num=img_num,
                                     s_t=s_t, t_a_ratio= t_a_ratio, selection=selection, q_cutoff= q_t)
        include_metadata_rating(img_list=img_list, q_file=q_path, t_a_ratio= t_a_ratio)
        summary = remove_folder_name(summary, folder)
        img_list = remove_folder_name(img_list, folder)
        print("Summary calculated")
        window.Refresh() if window else None
        toc = time.perf_counter()
        print(f"Process took: {toc - tic:0.2f} s")
        window.Refresh() if window else None
        if window_selection:
            window_selection.close()
        window_selection = make_win2()
        window_selection['-SUM_LIST'].update(summary)
        rest_list = natsort.natsorted(list(set(img_list) - set(summary)))
        window_selection['-REST_LIST'].update(rest_list)

    if event == '-SUM_LIST' or event == '-REST_LIST':  # A file was chosen from the listbox
        try:
            highlight = values[event][0]
            filename = os.path.join(folder, highlight)
            window['-TOUT-'].update(highlight)
            window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(900, 760)))
        except Exception:
            pass
    if event == '-MOVE_UP':
        updated = False
        window.write_event_value("-PARA_UPDATED", None)
        try:
            summary.append(highlight)
            ind = rest_list.index(highlight)
            rest_list.remove(highlight)
            window_selection['-SUM_LIST'].update(summary)
            window_selection['-REST_LIST'].update(rest_list)
            highlight = rest_list[ind]
            filename = os.path.join(folder, highlight)
            window['-TOUT-'].update(highlight)
            window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(900, 760)))
        except Exception:
            pass
    if event == '-MOVE_DOWN':
        updated = False
        window.write_event_value("-PARA_UPDATED", None)
        try:
            rest_list.append(highlight)
            ind = summary.index(highlight)
            summary.remove(highlight)
            window_selection['-SUM_LIST'].update(summary)
            window_selection['-REST_LIST'].update(rest_list)
            highlight = summary[ind]
            filename = os.path.join(folder, highlight)
            window['-TOUT-'].update(highlight)
            window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(900, 760)))
        except Exception:
            pass
    if event == "-COPY":
        folder_name = folder.split("/")[-1]
        save_folder = values["-COPY"] + "/Selected summary " + folder_name
        pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
        for img in summary:
            shutil.copy2(os.path.join(folder, img), save_folder)
    if event == "-UPDATE_PARA":
        window.perform_long_operation(lambda: update_parameters(summary=summary, img_list=img_list,
                                                                s_file=sim_path, q_file=q_path, nbrs=nbrs)
                                      , end_key="-PARA_UPDATED")
        updated = True
    if event == "-PARA_UPDATED":
        window['-UPDATE_PARA'].update(text="UPDATED" if updated else "Update parameters for automatic selection")
