__author__ = 'Lukáš Bartůněk'

import PySimpleGUI as sg
from quality_assessment import calculate_qualities
from similarity_assessment import calculate_similarities
from content_assessment import calculate_content
from summary_creation import select_summary
from utils import prepare_paths, prepare_img_list, remove_folder_name, copy_images, save_list, load_trained
import logistic_regression
import neural_network

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


def update_both_models(s, lst, s_file, q_file, nbrs):
    logistic_regression.update_parameters(s, lst, s_file, q_file)
    neural_network.update_model(s, lst, s_file, q_file, nbrs)


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
            sg.Slider((0, 20), orientation='h', s=(10, 15), default_value=10, resolution=1,
                      tooltip="Recommended: 10\nBased on maximum number of duplicates in a row", key="-NBRS"),
            sg.Push(),
            sg.Checkbox("Recalculate", key="-RECALC", enable_events=True)
        ],
        [sg.HSeparator(), ],
        [
            sg.Push(),
            sg.Text("Similarity threshold"),
            sg.Slider((0, 50), orientation='h', s=(10, 15), default_value=10, resolution=2,
                      key="-S_T"),
            sg.Push()
        ], [
            sg.Push(),
            sg.Text("Aesthetic quality "),
            sg.Slider((0, 100), orientation='h', s=(10, 15), default_value=50, resolution=2,
                      key="-T_A_RATIO"),
            sg.Text("Technical quality "),
            sg.Push(),
        ], [
            sg.Push(),
            sg.Text("Structural similarity "),
            sg.Slider((0, 100), orientation='h', s=(10, 15), default_value=50, resolution=2,
                      key="-S_C_RATIO"),
            sg.Text("Content similarity"),
            sg.Push(),
        ],
        [sg.HSeparator(), ],
        [
            sg.Push(),
            sg.Text("Selection based on:"),
            sg.Push(),
            sg.Button('Output size', size=(10, 2), button_color='white on green', key='-SELECTION_MODE'),
            sg.Push()
        ], [
            sg.Push(),
            sg.Text("Output size in percents"),
            sg.Push(),
            sg.Slider((0, 100), orientation='h', s=(10, 15), default_value=10, resolution=2,
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
            sg.Button("Selected parameters", key="-SUMM_MAN", size=(20, 1)),
            sg.Button("Preset parameters", key="-SUMM_REC", size=(20, 1)),
            sg.Push(),
        ], [
            sg.Push(),
            sg.Button("Automatic summary using logistic regression", key="-SUMM_AUTO_REG",
                      tooltip="Only selection based on quality", size=(40, 1)),
            sg.Push()
        ], [
            sg.Push(),
            sg.Button("Automatic summary using neural network", key="-SUMM_AUTO_NN",
                      tooltip="Only selection based on quality", size=(40, 1)),
            sg.Push()
        ],
        [sg.HSeparator(), ],
        [
            sg.Checkbox("Write metadata", key="-METADATA", enable_events=True, default=True),
            sg.Push(),
            sg.Checkbox("Use graphics card if possible", key="-CUDA", enable_events=True, default=True)
        ], [
            [sg.Output(size=(60, 6), key="-OUTPUT")],
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
            sg.In(key="-OUT_LIST", visible=False, enable_events=True),
            sg.FolderBrowse("Save selected list", size=(15, 2)),
            sg.Button("Close", key="-EXIT", size=(10, 2)),
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
auto_summ_nn = False
auto_summ_reg = False
updated = False
# Create an event loop
try:
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

        if event == '-SELECTION_MODE':
            selection = not selection
            window['-SELECTION_MODE'].update(text='Output size' if selection else 'Quality threshold',
                                 button_color='white on green' if selection else 'white on blue')
        if event == "-SUMM_REC" or event == "-SUMM_AUTO_REG" or event == "-SUMM_MAN" or event == "-SUMM_AUTO_NN":
            if event == "-SUMM_MAN":
                size = values["-SIZE"]
                s_t = values["-S_T"]
                t_a_ratio = values["-T_A_RATIO"] / 100
                s_c_ratio = values["-S_C_RATIO"] / 100
                q_t = values["-QUALITY_CUTOFF"]
            elif event == "-SUMM_REC":
                q_t, s_t, t_a_ratio, s_c_ratio, size = load_trained()
                window["-SIZE"].update(value=size)
                window["-S_T"].update(value=s_t)
                window["-T_A_RATIO"].update(value=t_a_ratio)
                window["-S_C_RATIO"].update(value=s_c_ratio)
                window["-QUALITY_CUTOFF"].update(value=q_t)
                window.refresh()
                t_a_ratio = t_a_ratio / 100
                s_c_ratio = s_c_ratio / 100
            elif event == "-SUMM_AUTO_REG":
                auto_summ_reg = True
                selection = not selection
                window['-SELECTION_MODE'].update(text='Output size' if selection else 'Quality threshold',
                                                 button_color='white on green' if selection else 'white on blue')
            elif event == "-SUMM_AUTO_NN":
                auto_summ_nn = True
                selection = not selection
                window['-SELECTION_MODE'].update(text='Output size' if selection else 'Quality threshold',
                                                 button_color='white on green' if selection else 'white on blue')
                window.refresh()
            if not folder:
                sg.popup("No folder selected")
            else:
                summ_create = True
                cuda = values["-CUDA"]
                window["-OUTPUT"].Update('')
                tic = time.perf_counter()
                print("Calculating quality...", end="   ")
                window.Refresh() if window else None
                window.perform_long_operation(lambda: calculate_qualities(lst=img_list, result_pth=q_path,
                                                                          model_pth="data/model.pth", cuda=cuda),
                                              end_key="-QUA_DONE")
        elif event == "-QUA_DONE" and summ_create:
            print("Quality calculated")
            window.Refresh() if window else None
            print("Creating content description...", end="   ")
            if not os.path.exists(c_path):
                recalc = True
            else:
                recalc = values["-RECALC"]
            window.Refresh() if window else None
            window.perform_long_operation(lambda: calculate_content(lst=img_list, result_pth=c_path, cuda=cuda)
                                          , end_key="-CON_DONE")

        elif event == "-CON_DONE" and summ_create:
            print("Content description created")
            window.Refresh() if window else None
            nbrs = int(values["-NBRS"])
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
            if auto_summ_reg:
                summary = logistic_regression.summary(lst=img_list, s_file=sim_path, q_file=q_path)
                weights = logistic_regression.load_weights()
                t_a_ratio = weights[0].item()
            elif auto_summ_nn:
                summary = neural_network.summary(lst=img_list, s_file=sim_path, q_file=q_path)
                t_a_ratio = 50
            else:
                summary = select_summary(sim_pth=sim_path, q_pth=q_path, size=size, num=img_num,
                                         s_t=s_t, t_a_ratio=t_a_ratio, s_c_ratio=s_c_ratio, selection=selection,
                                         q_cutoff= q_t)
            print("Summary calculated")
            window.Refresh() if window else None

            if values["-METADATA"]:
                print("Writing metadata...", end="   ")
                window.Refresh() if window else None
                include_metadata_rating(img_list=img_list, q_file=q_path, t_a_ratio=t_a_ratio)
                print("Metadata written")
                window.Refresh() if window else None

            summary = remove_folder_name(summary, folder)
            img_list_removed = remove_folder_name(img_list, folder)
            toc = time.perf_counter()
            print(f"Process took: {toc - tic:0.2f} s")
            window.Refresh() if window else None
            if window_selection:
                window_selection.close()
            window_selection = make_win2()
            window_selection['-SUM_LIST'].update(summary)
            rest_list = natsort.natsorted(list(set(img_list_removed) - set(summary)))
            window_selection['-REST_LIST'].update(rest_list)
            auto_summ_nn = False
            auto_summ_reg = False

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
            copy_images(summary=summary, folder=folder, dest_folder=values["-COPY"])
        if event == "-OUT_LIST":
            save_list(summary=summary, folder=folder, dest_folder=values["-OUT_LIST"])
        if event == "-UPDATE_PARA":
            if len(summary) == 0:
                sg.popup("Empty summary")
            window.perform_long_operation(lambda: update_both_models(s=summary, lst=img_list, s_file=sim_path,
                                                                     q_file=q_path, nbrs=nbrs)
                                          , end_key="-PARA_UPDATED")
            updated = True
        if event == "-PARA_UPDATED":
            window['-UPDATE_PARA'].update(text="UPDATED" if updated else "Update parameters for automatic selection")
except Exception as e:
    sg.popup_error_with_traceback(f"An error happened. Here is traceback:", e)
