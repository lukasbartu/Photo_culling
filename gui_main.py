__author__ = 'Lukáš Bartůněk'

import PySimpleGUI as sg
from quality_assessment import calculate_qualities
from similarity_assessment import calculate_similarities
from content_assessment import calculate_content
from summary_creation import select_summary, generate_preset_values
from utils import (prepare_paths, prepare_img_list, remove_folder_name,
                   copy_images, save_list, load_trained)
from metadata_creation import include_metadata_rating
import logical_approximation
import neural_network

import time
import os
import PIL
import PIL.Image
import io
import base64
import natsort


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
        except Exception:
            data_bytes_io = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(data_bytes_io)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height / cur_height, new_width / cur_width)
        img = img.resize((int(cur_width * scale), int(cur_height * scale)), PIL.Image.LANCZOS)
    with io.BytesIO() as bio:
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()


# PysimpleGUI demo
def ColumnFixedSize(layout, size=(None, None), *args, **kwargs):
    # An addition column is needed to wrap the column with the Sizers because the colors will not be set on the space the sizers take
    return sg.Column([[sg.Column([[sg.Sizer(0,size[1]-1), sg.Column([[sg.Sizer(size[0]-2,0)]] + layout, *args, **kwargs, pad=(0,0))]], *args, **kwargs)]],pad=(0,0))


def update_both_models(s, lst, s_file, q_file):
    logical_approximation.update_parameters(s, lst, s_file, q_file)
    neural_network.update_model(s, lst, s_file, q_file)


def reset_models():
    logical_approximation.reset_model()
    neural_network.reset_model()


manual_col = [
    [
        sg.Push(),
        sg.Text("Quality threshold"),
        sg.Push(),
    ], [
        sg.Push(),
        sg.Text("Low"),
        sg.Slider((0, 100), orientation='h', s=(10, 15), default_value=50, resolution=1, key="-QUALITY_CUTOFF"),
        sg.Text("High"),
        sg.Push()
    ], [
        sg.HSeparator()
    ], [
        sg.Push(),
        sg.Text("Similarity threshold"),
        sg.Push(),
    ], [
        sg.Push(),
        sg.Text("Low"),
        sg.Slider((0, 100), orientation='h', s=(10, 15), default_value=10, resolution=1, key="-S_T"),
        sg.Text("High"),
        sg.Push()
    ], [
        sg.HSeparator(),
    ], [
        sg.Push(),
        sg.Text("Quality priority"),
        sg.Push(),
    ], [
        sg.Push(),
        sg.Text("Aesthetic quality"),
        sg.Slider((0, 100), orientation='h', s=(10, 15), default_value=50, resolution=1,
                  key="-T_A_RATIO"),
        sg.Text("Technical quality"),
        sg.Push(),
    ], [
        sg.HSeparator(),
    ], [
        sg.Push(),
        sg.Text("Similarity priority"),
        sg.Push(),
    ], [
        sg.Push(),
        sg.Text("Content similarity"),
        sg.Slider((0, 100), orientation='h', s=(10, 15), default_value=50, resolution=1,
                  key="-S_C_RATIO"),
        sg.Text("Structural similarity"),
        sg.Push(),
    ]
]

automatic_col = [
    [
        sg.Push(),
        sg.Text("Quality threshold"),
        sg.Push(),
    ], [
        sg.Push(),
        sg.Button("Low", key="-L_QUA_THRESHOLD", size=(12,1)),
        sg.Button("Medium", key="-B_QUA_THRESHOLD",button_color="white on green",size=(12,1)),
        sg.Button("High", key="-H_QUA_THRESHOLD",size=(12,1)),
        sg.Push()
    ], [
        sg.HSeparator()
    ], [
        sg.Push(),
        sg.Text("Similarity threshold"),
        sg.Push(),
    ], [
        sg.Push(),
        sg.Button("Barely similar", key="-L_SIM_THRESHOLD",size=(12,1)),
        sg.Button("Similar", key="-B_SIM_THRESHOLD",button_color="white on green",size=(12,1)),
        sg.Button("Near duplicates", key="-H_SIM_THRESHOLD",size=(12,1)),
        sg.Push()
    ], [
        sg.HSeparator(),
    ], [
        sg.Push(),
        sg.Text("Quality priority"),
        sg.Push(),
    ], [
        sg.Push(),
        sg.Button("Aesthetic ", key="-L_QUALITY",size=(12,1)),
        sg.Button("Balanced", key="-B_QUALITY",button_color="white on green",size=(12,1)),
        sg.Button("Technical ", key="-H_QUALITY",size=(12,1)),
        sg.Push(),
    ], [
        sg.HSeparator(),
    ], [
        sg.Push(),
        sg.Text("Similarity priority"),
        sg.Push(),
    ], [
        sg.Push(),
        sg.Button("Content", key="-L_SIMILARITY",size=(12,1)),
        sg.Button("Balanced", key="-B_SIMILARITY",button_color="white on green",size=(12,1)),
        sg.Button("Structural", key="-H_SIMILARITY",size=(12,1)),
        sg.Push(),
    ]
]

man_auto =[
    [
        sg.Column(manual_col, key='-COL_MANUAL', visible=False, element_justification='c'),
        sg.Column(automatic_col, key='-COL_AUTO', visible=True, element_justification='c'),
    ]
]

man_auto_sel = [
    [
        sg.Push(),
        sg.Text("SETTINGS:"),
        sg.Button("Simple", key="-SWITCH_AUTO", button_color="white on green"),
        sg.Button("Advanced", key="-SWITCH_MANUAL"),
        sg.Button("Recommended", key="-SWITCH_REC"),
        sg.Push()
    ], [
        sg.HSeparator()
    ], [
        ColumnFixedSize(man_auto, size=(10, 300), element_justification='c')
    ]
]

man_auto_sel_help = [
    [sg.Column(man_auto_sel, key="-MAN_AUTO", visible=False, element_justification='c')]
]

size_col = [
    [
        sg.Push(),
        sg.Text("Output size in percent"),
        sg.Slider((0, 100), orientation='h', s=(10, 15), default_value=10, resolution=1,
                  key="-SIZE"),
        sg.Push(),
    ]
]

bottom_col = [
    [
        sg.HSeparator()
    ], [
        sg.Push(),
        sg.Button("Generate", key="-GENERATE", tooltip="Generates selection based on method selected by user."),
        sg.Push()
    ], [
        sg.Checkbox("Write metadata", key="-METADATA", enable_events=True, default=False,
                    tooltip="Writes relative image quality of each picture into its metadata."),
        sg.Push(),
        sg.Checkbox("Use graphics card if possible", key="-CUDA", enable_events=True, default=True,
                    tooltip="Some systems without graphics card need uncheck this to avoid error message.")
    ], [
        sg.Output(size=(55, 6), key="-OUTPUT")
    ]
]

layout1 = [
    [
        sg.Text("Selection settings", size=(15, 4), font=10)
    ], [
        sg.Push(),
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(tooltip="Select directory/folder with image sequence for culling."),
        sg.Push()
    ], [
        sg.Text("Choose selection method")
    ], [
        sg.Button("Conditional statements", key="-CS_BUTTON", size=(12, 2),
                  tooltip="This method creates image selection by using if-else statements to formulate the selection "
                          "logic. \n It allows user to specify setting and influence the selection process. \n "
                          "It offers simple setting, advanced settings and settings that we recommend. \n"
                          "Recommended setting is based on our experiments but could not be well suited for all uses."),
        sg.Button("Logical approximation", key="-LR_BUTTON", size=(12, 2),
                  tooltip="This method creates image selection by using approximation of selection logic.\n"
                          "It is fully automatic and does not need user settings to work.\n It uses setting that can"
                          "be updated for better approximation of user preference."),
        sg.Button("Neural network", key="-NN_BUTTON", size=(12, 2),
                  tooltip="This method uses neural networks model that was trained to generate selections. \n"
                          "It is fully automatic and does not need user settings to work. \nIt can be updated for"
                          "better approximation of user preference.")
    ], [
        sg.HSeparator(),
    ], [
        sg.Push(),
        sg.Text('Specify size of output?'),
        sg.Push(),
        sg.Button('Specify', size=(10, 2), button_color='white on green', key='-SELECTION_MODE'),
        sg.Push()
    ], [
        sg.Column(size_col, key="-SELECTION_TEXT")
    ], [
        sg.HSeparator(),
    ], [
        ColumnFixedSize(man_auto_sel_help, size=(10, 350), element_justification='c')
    ], [
        sg.Column(bottom_col, vertical_alignment='bottom', element_justification='c')
    ]
]

sum_but_col = [
    [
        sg.Button("↑", key="-SUM_BUTTON_UP", size=(5, 2)),
    ], [
        sg.Button("Deselect image", key="-MOVE_DOWN", size=(10, 2)),
    ], [
        sg.Button("↓", key="-SUM_BUTTON_DOWN", size=(5, 2)),
    ]
]

rest_but_col = [
    [
        sg.Button("↑", key="-REST_BUTTON_UP", size=(5, 2)),
    ], [
        sg.Button("Select image", key="-MOVE_UP", size=(10, 2)),
    ], [
        sg.Button("↓", key="-REST_BUTTON_DOWN", size=(5, 2)),
    ]
]

left_col = [
    [
        sg.Text("Selected images"),
    ], [
        sg.Listbox(values=[], enable_events=True, size=(40, 25), key='-SUM_LIST'),
        sg.Column(sum_but_col,element_justification='c')
    ], [
        sg.Text("Not selected images")
    ], [
        sg.Listbox(values=[], enable_events=True, size=(40, 25), key='-REST_LIST'),
        sg.Column(rest_but_col,element_justification='c')
    ]
]

images_col = [
    [
        sg.Text('You choose from the list:')
    ], [
        sg.Push(),
        sg.Text(key='-TOUT-'),
        sg.Push()
    ], [
        sg.Image(key='-IMAGE-', size=(900, 760))
    ]
]

layout2 = [
    [
        sg.Column(layout1, element_justification='c'),
        sg.VSeparator(),
        sg.Column(left_col, element_justification='c'),
        sg.VSeparator(),
        sg.Column(images_col, element_justification='c')
    ], [
        sg.Push(),
        sg.Button("Reset parameters for automatic selection to default values", key="-RESET_PARA",
                  size=(25, 2)),
        sg.Push(),
        sg.Button("Update parameters for automatic selection", key="-UPDATE_PARA", size=(20, 2)),
        sg.In(key="-COPY", visible=False, enable_events=True),
        sg.FolderBrowse("Copy selection into folder", size=(15, 2)),
        sg.In(key="-OUT_LIST", visible=False, enable_events=True),
        sg.FolderBrowse("Save selected list", size=(15, 2)),
        sg.Button("EXIT", key="-EXIT", size=(10, 2)),
    ]
]

window = sg.Window("Photo selector", layout2, finalize=True)

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
size_based = True
summary = []
rest_list = []
selection_mode = 2
method_mode = 0
highlight = None
sum_highlight = True
highlight_idx = 0
qua_preset = 2
sim_preset = 2
tar_preset = 2
scr_preset = 2
tic = 0
q_t, s_t, t_a_ratio, s_c_ratio, = 0, 0, 0, 0
# Create an event loop
try:
    while True:
        window, event, values = sg.read_all_windows()
        if event == sg.WIN_CLOSED or event == '-EXIT':
            window.close()
            break
        if event == "-CS_BUTTON":
            window['-MAN_AUTO'].update(visible=True)
            method_mode = 1
            window['-CS_BUTTON'].update(button_color='white on green' if method_mode == 1
                                        else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-LR_BUTTON'].update(button_color='white on green' if method_mode == 2
                                        else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-NN_BUTTON'].update(button_color='white on green' if method_mode == 3
                                        else (sg.theme_text_color(), sg.theme_button_color_background()))
        elif event == "-LR_BUTTON":
            window['-MAN_AUTO'].update(visible=False)
            method_mode = 2
            window['-CS_BUTTON'].update(button_color='white on green' if method_mode == 1
                                        else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-LR_BUTTON'].update(button_color='white on green' if method_mode == 2
                                        else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-NN_BUTTON'].update(button_color='white on green' if method_mode == 3
                                        else (sg.theme_text_color(), sg.theme_button_color_background()))
        elif event == "-NN_BUTTON":
            window['-MAN_AUTO'].update(visible=False)
            method_mode = 3
            window['-CS_BUTTON'].update(button_color='white on green' if method_mode == 1
                                        else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-LR_BUTTON'].update(button_color='white on green' if method_mode == 2
                                        else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-NN_BUTTON'].update(button_color='white on green' if method_mode == 3
                                        else (sg.theme_text_color(), sg.theme_button_color_background()))
        if event == "-SWITCH_MANUAL":
            selection_mode = 1
            window[f"-COL_MANUAL"].update(visible=True)
            window[f"-COL_AUTO"].update(visible=False)
            window['-SWITCH_MANUAL'].update(button_color='white on green' if selection_mode == 1
                                            else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-SWITCH_AUTO'].update(button_color='white on green' if selection_mode == 2
                                          else (sg.theme_text_color(), sg.theme_button_color_background()))
        elif event == "-SWITCH_AUTO":
            selection_mode = 2
            window[f"-COL_MANUAL"].update(visible=False)
            window[f"-COL_AUTO"].update(visible=True)
            window['-SWITCH_MANUAL'].update(button_color='white on green' if selection_mode == 1
                                            else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-SWITCH_AUTO'].update(button_color='white on green' if selection_mode == 2
                                          else (sg.theme_text_color(), sg.theme_button_color_background()))
        elif event == "-SWITCH_REC":
            selection_mode = 1
            window[f"-COL_MANUAL"].update(visible=True)
            window[f"-COL_AUTO"].update(visible=False)
            window['-SWITCH_MANUAL'].update(button_color='white on green' if selection_mode == 1
                                            else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-SWITCH_AUTO'].update(button_color='white on green' if selection_mode == 2
                                          else (sg.theme_text_color(), sg.theme_button_color_background()))
            q_t, s_t, t_a_ratio, s_c_ratio = load_trained()
            window["-QUALITY_CUTOFF"].update(value=q_t)
            window["-S_T"].update(value=s_t)
            window["-T_A_RATIO"].update(value=t_a_ratio)
            window["-S_C_RATIO"].update(value=s_c_ratio)
            window.refresh()
        if (event == "-L_QUA_THRESHOLD" or event == "-B_QUA_THRESHOLD" or event == "-H_QUA_THRESHOLD"
                or event == "-SWITCH_AUTO"):
            if event == "-L_QUA_THRESHOLD":
                qua_preset = 1
            elif event == "-B_QUA_THRESHOLD":
                qua_preset = 2
            elif event == "-H_QUA_THRESHOLD":
                qua_preset = 3
            window['-L_QUA_THRESHOLD'].update(button_color='white on green' if qua_preset == 1
            else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-B_QUA_THRESHOLD'].update(button_color='white on green' if qua_preset == 2
            else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-H_QUA_THRESHOLD'].update(button_color='white on green' if qua_preset == 3
            else (sg.theme_text_color(), sg.theme_button_color_background()))

        if (event == "-L_SIM_THRESHOLD" or event == "-B_SIM_THRESHOLD" or event == "-H_SIM_THRESHOLD"
                or event == "-SWITCH_AUTO"):
            if event == "-L_SIM_THRESHOLD":
                sim_preset = 1
            elif event == "-B_SIM_THRESHOLD":
                sim_preset = 2
            elif event == "-H_SIM_THRESHOLD":
                sim_preset = 3
            window['-L_SIM_THRESHOLD'].update(button_color='white on green' if sim_preset == 1
            else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-B_SIM_THRESHOLD'].update(button_color='white on green' if sim_preset == 2
            else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-H_SIM_THRESHOLD'].update(button_color='white on green' if sim_preset == 3
            else (sg.theme_text_color(), sg.theme_button_color_background()))
        if (event == "-L_QUALITY" or event == "-B_QUALITY" or event == "-H_QUALITY"
                or event == "-SWITCH_AUTO"):
            if event == "-L_QUALITY":
                tar_preset = 1
            elif event == "-B_QUALITY":
                tar_preset = 2
            elif event == "-H_QUALITY":
                tar_preset = 3
            window['-L_QUALITY'].update(button_color='white on green' if tar_preset == 1
            else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-B_QUALITY'].update(button_color='white on green' if tar_preset == 2
            else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-H_QUALITY'].update(button_color='white on green' if tar_preset == 3
            else (sg.theme_text_color(), sg.theme_button_color_background()))
        if (event == "-L_SIMILARITY" or event == "-B_SIMILARITY" or event == "-H_SIMILARITY"
                or event == "-SWITCH_AUTO"):
            if event == "-L_SIMILARITY":
                scr_preset = 1
            elif event == "-B_SIMILARITY":
                scr_preset = 2
            elif event == "-H_SIMILARITY":
                scr_preset = 3
            window['-L_SIMILARITY'].update(button_color='white on green' if scr_preset == 1
            else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-B_SIMILARITY'].update(button_color='white on green' if scr_preset == 2
            else (sg.theme_text_color(), sg.theme_button_color_background()))
            window['-H_SIMILARITY'].update(button_color='white on green' if scr_preset == 3
            else (sg.theme_text_color(), sg.theme_button_color_background()))
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            sim_path, q_path, c_path = prepare_paths(folder)
            img_list, img_num = prepare_img_list(folder)
        if event == '-SELECTION_MODE':
            size_based = not size_based
            window['-SELECTION_MODE'].update(text='Specify' if size_based else 'No limit',
                                             button_color='white on green' if size_based else
                                             (sg.theme_text_color(), sg.theme_button_color_background()))
            window["-SELECTION_TEXT"].update(visible=True if size_based else False)
        if event == "-GENERATE":
            if not folder:
                sg.popup("No folder selected")
            elif method_mode == 0:
                sg.popup("Choose selection method")
            else:
                size = values["-SIZE"]
                if selection_mode == 1:
                    q_t = values["-QUALITY_CUTOFF"]
                    s_t = values["-S_T"]
                    t_a_ratio = values["-T_A_RATIO"]/100
                    s_c_ratio = values["-S_C_RATIO"]/100
                else:
                    q_t, s_t, t_a_ratio, s_c_ratio = generate_preset_values(qua_preset, sim_preset, tar_preset,
                                                                            scr_preset, q_path, sim_path)
                summ_create = True
                cuda = values["-CUDA"]
                window["-OUTPUT"].Update('')
                tic = time.perf_counter()
                print("Calculating quality...", end="   ")
                window.Refresh() if window else None
                window.perform_long_operation(lambda: calculate_qualities(lst=img_list, result_pth=q_path, cuda=cuda),
                                              end_key="-QUA_DONE")
        elif event == "-QUA_DONE" and summ_create:
            print("Quality calculated")
            window.Refresh() if window else None
            print("Creating content description...", end="   ")
            if not os.path.exists(c_path):
                recalc = True
            else:
                recalc = False
            window.Refresh() if window else None
            window.perform_long_operation(lambda: calculate_content(lst=img_list, result_pth=c_path, cuda=cuda),
                                          end_key="-CON_DONE")

        elif event == "-CON_DONE" and summ_create:
            print("Content description created")
            window.Refresh() if window else None
            window.Refresh() if window else None
            print("Calculating similarities...", end="   ")
            window.Refresh() if window else None

            window.perform_long_operation(lambda: calculate_similarities(lst=img_list, result_pth=sim_path, num=img_num,
                                                                         nbrs=20, content_pth=c_path, recalc=recalc),
                                          end_key="-SIM_DONE")

        elif event == "-SIM_DONE" and summ_create:
            print("Similarities calculated")
            window.Refresh() if window else None
            print("Selecting summary of photos...", end="   ")
            window.Refresh() if window else None
            output_size = int(img_num * (size / 100))
            if method_mode == 2:
                summary = logical_approximation.summary(lst=img_list, s_file=sim_path, q_file=q_path,
                                                        size_based=size_based, output_size=output_size)
                weights = logical_approximation.load_weights()
                t_a_ratio = weights[0].item()
            elif method_mode == 3:
                summary = neural_network.summary(lst=img_list, s_file=sim_path, q_file=q_path, size_based=size_based,
                                                 output_size=output_size)
                t_a_ratio = 50
            elif method_mode == 1:
                summary = select_summary(sim_pth=sim_path, q_pth=q_path, size=size, num=img_num,
                                         s_t=s_t, t_a_ratio=t_a_ratio, s_c_ratio=s_c_ratio, size_based=size_based,
                                         q_cutoff=q_t)
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
            window['-SUM_LIST'].update(summary)
            highlight = summary[0]
            window['-TOUT-'].update(highlight)
            highlight_idx = summary.index(highlight)
            window["-SUM_LIST"].update(set_to_index=highlight_idx)
            filename = os.path.join(folder, highlight)
            sum_highlight = True
            window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(900, 760)))
            rest_list = natsort.natsorted(list(set(img_list_removed) - set(summary)))
            window['-REST_LIST'].update(rest_list)
            auto_summ_nn = False
            auto_summ_reg = False
        if event == '-SUM_LIST' or event == '-REST_LIST':  # A file was chosen from the listbox
            if event == "-REST_LIST":
                sum_highlight = False
            else:
                sum_highlight = True
            try:
                highlight = values[event][0]
                highlight_idx = window[event].get_indexes()[0]
                filename = os.path.join(folder, highlight)
                window["-SUM_LIST"].update(set_to_index=[])
                window["-REST_LIST"].update(set_to_index=[])
                window[event].update(set_to_index=highlight_idx)
                window['-TOUT-'].update(highlight)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(900, 760)))
            except Exception:
                pass
        if event == '-MOVE_UP':
            updated = False
            window.write_event_value("-PARA_UPDATED", None)
            try:
                summary.append(highlight)
                summary = natsort.natsorted(summary)
                rest_list.remove(highlight)
                window['-SUM_LIST'].update(summary)
                window['-REST_LIST'].update(rest_list)
                highlight_idx = summary.index(highlight)
                highlight = summary[highlight_idx]
                window["-REST_LIST"].update(set_to_index=[])
                window["-SUM_LIST"].update(set_to_index=highlight_idx,scroll_to_index=max(highlight_idx-5,0))
                sum_highlight = True
                filename = os.path.join(folder, highlight)
                window['-TOUT-'].update(highlight)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(900, 760)))
            except Exception:
                pass
        elif event == '-MOVE_DOWN':
            updated = False
            window.write_event_value("-PARA_UPDATED", None)
            try:
                rest_list.append(highlight)
                rest_list = natsort.natsorted(rest_list)
                summary.remove(highlight)
                window['-SUM_LIST'].update(summary)
                window['-REST_LIST'].update(rest_list)
                highlight_idx = rest_list.index(highlight)
                highlight = rest_list[highlight_idx]
                window["-REST_LIST"].update(set_to_index=highlight_idx, scroll_to_index=max(highlight_idx-5,0))
                window["-SUM_LIST"].update(set_to_index=[])
                sum_highlight = False
                filename = os.path.join(folder, highlight)
                window['-TOUT-'].update(highlight)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(900, 760)))
            except Exception:
                pass
        if event == "-SUM_BUTTON_DOWN":
            if sum_highlight and highlight_idx+1 < len(summary):
                highlight_idx = highlight_idx + 1
                highlight = summary[highlight_idx]
                window["-SUM_LIST"].update(set_to_index=highlight_idx)
                filename = os.path.join(folder, highlight)
                window['-TOUT-'].update(highlight)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(900, 760)))
        elif event == "-SUM_BUTTON_UP":
            if sum_highlight and highlight_idx-1 >= 0:
                highlight_idx = highlight_idx - 1
                highlight = summary[highlight_idx]
                window["-SUM_LIST"].update(set_to_index=highlight_idx)
                filename = os.path.join(folder, highlight)
                window['-TOUT-'].update(highlight)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(900, 760)))
        if event == "-REST_BUTTON_DOWN":
            if not sum_highlight and highlight_idx+1 < len(rest_list):
                highlight_idx = highlight_idx + 1
                highlight = rest_list[highlight_idx]
                window["-REST_LIST"].update(set_to_index=highlight_idx)
                filename = os.path.join(folder, highlight)
                window['-TOUT-'].update(highlight)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(900, 760)))
        elif event == "-REST_BUTTON_UP":
            if not sum_highlight and highlight_idx-1 >= 0:
                highlight_idx = highlight_idx - 1
                highlight = rest_list[highlight_idx]
                window["-REST_LIST"].update(set_to_index=highlight_idx)
                filename = os.path.join(folder, highlight)
                window['-TOUT-'].update(highlight)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(900, 760)))
        if event == "-COPY":
            copy_images(summary=summary, folder=folder, dest_folder=values["-COPY"])
        if event == "-OUT_LIST":
            save_list(summary=summary, folder=folder, dest_folder=values["-OUT_LIST"])
        if event == "-UPDATE_PARA":
            if len(summary) == 0:
                sg.popup("Empty summary")
            window.perform_long_operation(lambda: update_both_models(s=summary, lst=img_list_removed, s_file=sim_path,
                                                                     q_file=q_path),
                                          end_key="-PARA_UPDATED")
            window['-UPDATE_PARA'].update(text="Updating...")
            updated = True
        if event == "-PARA_UPDATED":
            window['-UPDATE_PARA'].update(text="UPDATED" if updated else "Update parameters for automatic selection")
        if event == "-RESET_PARA":
            if sg.popup_yes_no("Are you sure you want to reset automatic parameters?", title="Confirmation"):
                reset_models()

except Exception as e:
    sg.popup_error_with_traceback(f"An error happened. Here is traceback:", e)
