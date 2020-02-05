import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from plot_utils import DiscreteSlider
import math
import numpy as np
import matplotlib.animation as animation
import pandas as pd
import cv2
import sys
import os


def read_files(path, file_name):
    """
    Read the ego, dynamic and static file for one recording
    :param path: the path where three csv files saved
    :param file_name: the prefix of current file name
    :return: three dataframes contain ego, dynamic and static respectively
    """

    if os.path.exists(
            r'{}\{}_dynamic.csv'.format(path, file_name)) and os.path.exists(
        r'{}\{}_static.csv'.format(path, file_name)) and os.path.exists(
        r'{}\{}_ego.csv'.format(path, file_name)):
        with open(r'{}\{}_dynamic.csv'.format(path, file_name)) as tmp_dynamic:
            dynamic_csv = pd.read_csv(tmp_dynamic)
            print('Dynamic csv file found')
        with open(r'{}\{}_static.csv'.format(path, file_name)) as tmp_static:
            static_csv = pd.read_csv(tmp_static)
            print('Static csv file found')
        with open(r'{}\{}_ego.csv'.format(path, file_name)) as tmp_ego:
            ego_csv = pd.read_csv(tmp_ego)
            print('Ego csv file found')
        return ego_csv, dynamic_csv, static_csv

    else:
        print('No available data')
        sys.exit(0)


def process_dynamic(dynamic_df):
    """
    Process df of dynamic info
    :param dynamic_df: df of dynamic info
    :return: a dict which contains all the objects info in recording
    """
    grouped_dynamic = dynamic_df.groupby(['obj_id'], sort=False)
    objs_dynamic = [None] * grouped_dynamic.ngroups
    current_obj = 0
    for obj_id, rows in grouped_dynamic:
        objs_dynamic[current_obj] = {'obj_id': np.int64(obj_id) - 1,
                                     # minus one to ensure the obj id is consistent with the obj id in static recording
                                     'frame': rows['frame'].values,
                                     'pos_x': rows['pos_x'].values,
                                     'pos_y': rows['pos_y'].values,
                                     'speed_x': rows['speed_x'].values,
                                     'speed_y': rows['speed_y'].values,
                                     'cut_in_left': rows['cut_in_left'].values,
                                     'cut_in_right': rows['cut_in_right'].values}
        current_obj += 1
    return objs_dynamic


def obj_bounding_shape(obj_class,
                       # length,
                       width):
    """
    This file return a stable shape of objects according to its type
    :param obj_class: class of the objects
    :param length: the length of the object saved in static df file
    :param width: the length of the object saved in static df file
    :return: a consistent length and width of given object
    """
    # Given the visualized object a shape
    if obj_class == 'Truck':
        length = 12
    elif obj_class == 'Motorcycle':
        length = 2
    elif obj_class == 'Car':
        length = 5
    else:
        length = 3

    if width == 0:
        if obj_class == 'Truck':
            width = 4
        elif obj_class == 'Motorcycle':
            width = 1
        elif obj_class == 'Car':
            width = 2
        else:
            width = 3

    return np.transpose(np.array([length, width]))


def process_static(static_df):
    """
    Process df of static info
    :param static_df: df of static info
    :return: a dict which contains all the objects info in recording
    """
    objs_static = {}
    for obj_id in range(static_df.shape[0]):
        objs_static[obj_id] = {'obj_id': obj_id,
                               'bounding_shape': obj_bounding_shape(static_df['obj_class'][obj_id],
                                                                    # static_df['length'][obj_id],
                                                                    static_df['width'][obj_id]),
                               'obj_class': static_df['obj_class'][obj_id],
                               'initial_frame': static_df['initial_frame'][obj_id],
                               'total_frames': static_df['total_frames'][obj_id]}
    return objs_static


def video(path):
    """
    Extract video from file
    :param path: path to video
    :return: openCV object and total frames of video
    """
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count, cap


class VisualizationPlot(object):

    def __init__(self, ego, dynamic, static, video_bool, *cap):
        self.ego_df = ego
        self.objs_dynamic_dict = dynamic
        self.objs_static_dict = static
        self.current_frame = 0
        self.maximum_frames = ego.shape[0] - 1
        self.plotted_objects = []
        self.changed_button = False
        self.anim_running = True
        self.text = ''
        self.steering_ang = True if 'steering_ang' in self.ego_df.columns else False
        self.video_bool = video_bool
        self.lane_change_count = 0
        self.lane_change_left = False
        self.lane_change_right = False
        self.cut_in_count = 0
        # self.cut_in_count = 0
        # self.cut_in_left_flag = False
        # self.cut_in_right_flag = False
        if self.video_bool:
            self.cap_count = int(cap[0])
            self.cap = cap[1]
        # Create figure and axes
        self.fig, self.ax = plt.subplots(1, 1)

        self.t = self.ax.text(21, 96, self.text, bbox={'facecolor': 'wheat', 'alpha': 0.5, 'boxstyle': 'round'})
        # self.fig.set_size_inches(32, 32)
        axes = plt.gca()
        # plt.axis('scaled')
        axes.set_xlim([-40, 40])
        axes.set_ylim([-3, 150])

        # Initialize the plot with the bounding boxes of the first frame
        self.update_figure()

        ax_color = 'lightgoldenrodyellow'
        # Define axes for the widgets
        self.ax_button_play = self.fig.add_axes([0, 0.01, 0.11, 0.05])  # Play button
        self.ax_button_previous = self.fig.add_axes([0.11, 0.01, 0.1, 0.05])  # Previous button
        self.ax_slider = self.fig.add_axes([0.3, 0.02, 0.41, 0.03], facecolor=ax_color)  # Slider
        self.ax_button_next = self.fig.add_axes([0.8, 0.01, 0.1, 0.05])  # Next button
        self.ax_button_next5 = self.fig.add_axes([0.9, 0.01, 0.1, 0.05])  # Next x5 button
        # Define the widgets

        self.frame_slider = DiscreteSlider(self.ax_slider, 'Frame', 1, self.maximum_frames, valinit=self.current_frame,
                                           valfmt='%s')
        self.button_play = Button(self.ax_button_play, 'Play/Stop')
        self.button_previous = Button(self.ax_button_previous, 'Prev. x10')
        self.button_next = Button(self.ax_button_next, 'Next')
        self.button_next25 = Button(self.ax_button_next5, 'Next x25')

        # Define the callbacks for the widgets' actions
        self.frame_slider.on_changed(self.update_slider)
        self.button_previous.on_clicked(self.update_button_previous5)
        self.button_next.on_clicked(self.update_button_next)
        self.button_play.on_clicked(self.update_button_play)
        self.button_next25.on_clicked(self.update_button_next25)

        self.ax.set_autoscale_on(False)
        # Interval unit: ms
        self.anim = animation.FuncAnimation(self.fig, self.play,
                                            interval=0)

    def play_video(self, cap_frame):
        """
        play the video based on currend frame
        :param cap_frame: current frame in the video
        """
        # Plus 35 to synchronize video and recording
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, cap_frame+30)
        _, frame = self.cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    def update_slider(self, value):
        if not self.changed_button:
            self.current_frame = value
            self.remove_patches()
            self.update_figure()
            self.fig.canvas.draw_idle()
        self.changed_button = False

    def update_button_next(self, _):
        if self.current_frame < self.maximum_frames:
            self.current_frame = self.current_frame + 1
            self.changed_button = True
            self.trigger_update()
        else:
            print("There are no frames available with an index higher than {}.".format(self.maximum_frames))  #

    def update_button_next25(self, _):
        if self.current_frame + 25 <= self.maximum_frames:
            self.current_frame = self.current_frame + 25
            self.changed_button = True
            self.trigger_update()
        else:
            print("There are no frames available with an index higher than {}.".format(self.maximum_frames))

    def update_button_previous5(self, _):
        if self.current_frame - 10 > 1:
            self.current_frame = self.current_frame - 10
            self.changed_button = True
            self.trigger_update()
        else:
            print("There are no frames available with an index lower than 1.")

    def update_button_play(self, event):
        if self.anim_running:
            self.anim.event_source.stop()
            self.anim_running = False
        else:
            self.anim.event_source.start()
            self.anim_running = True

    def play(self, frame):
        if self.current_frame < self.maximum_frames:
            self.current_frame = self.current_frame + 1
            self.changed_button = True
            self.trigger_update()
        else:
            print("There are no frames available with an index lower than 1.")

    def trigger_update(self):
        self.remove_patches()
        self.update_figure()
        self.frame_slider.update_val_external(self.current_frame)
        self.fig.canvas.draw_idle()

    def update_figure(self):
        # Style of the different objects that are visualized
        # Line color: black
        # ego color: blue
        # other cars color: cyan
        # left cut in color: green
        # right cut in color: red
        # left lane change: yellow
        # right lane change: pink

        # Play the video
        if self.video_bool:
            self.play_video(int(round(self.current_frame / self.maximum_frames * self.cap_count)))

        line_style = dict(color="#141310", linewidth=1, zorder=10)
        rect_style = dict(facecolor="#0b2299", fill=True, edgecolor="k", zorder=19)
        cut_in_left_style = dict(facecolor="#19e60e", fill=True, edgecolor="k", zorder=19)
        cut_in_right_style = dict(facecolor="#fa1111", fill=True, edgecolor="k", zorder=19)

        line_change_left_style = dict(facecolor="#f7f01b", fill=True, edgecolor="k", zorder=19)
        line_change_right_style = dict(facecolor="#e616f5", fill=True, edgecolor="k", zorder=19)
        ego_style = dict(facecolor="#0e7de6", fill=True, edgecolor="k", zorder=19)

        # Plot the bounding boxes of current objects and lines
        plotted_objects = []
        for obj in self.objs_dynamic_dict:
            obj_id = obj['obj_id']
            obj_static = self.objs_static_dict[obj_id]
            initial_frame = obj_static['initial_frame']
            total_frames = obj_static['total_frames']
            if initial_frame <= self.current_frame < initial_frame + total_frames:
                current_frame = self.current_frame - initial_frame
                pos_x = obj['pos_x'][current_frame]
                pos_y = obj['pos_y'][current_frame]
                length = obj_static['bounding_shape'][0]
                width = obj_static['bounding_shape'][1]

                if obj['cut_in_left'][current_frame] == 1 or obj['cut_in_right'][current_frame] == 1:
                    self.cut_in_count = 25
                # if obj['cut_in_left'][current_frame] == 1:
                #     self.cut_in_flag
                #     rect = plt.Rectangle((-pos_y - width / 2, pos_x - length / 2), width,
                #                          length, **cut_in_left_style)
                # elif obj['cut_in_right'][current_frame] == 1:
                #     rect = plt.Rectangle((-pos_y - width / 2, pos_x - length / 2), width,
                #                          length, **cut_in_right_style)
                # else:
                #     rect = plt.Rectangle((-pos_y - width / 2, pos_x - length / 2), width,
                #                          length, **rect_style)
                rect = plt.Rectangle((-pos_y - width / 2, pos_x - length / 2), width,
                                     length, **rect_style)
                self.ax.add_patch(rect)
                plotted_objects.append(rect)

        # Add ego line
        ego_current = self.ego_df.loc[self.current_frame]
        line_left_x, line_left_y = self.get_left_line(ego_current)
        if isinstance(line_left_x, np.ndarray):
            line_right_x, line_right_y = self.get_right_line(ego_current)
            if isinstance(line_right_x, np.ndarray):
                left_line = self.ax.plot(-line_left_y, line_left_x, **line_style)
                right_line = self.ax.plot(-line_right_y, line_right_x, **line_style)
                plotted_objects.append(left_line)
                plotted_objects.append(right_line)

        # Add ego info window
        self.text = self.display_info(ego_current)
        self.t.set_text(self.text)

        # Add ego vehicle
        if ego_current['lane_change_left'] == 1:
            self.lane_change_left = True
            self.lane_change_count = 25
        if ego_current['lane_change_right'] == 1:
            self.lane_change_right = True
            self.lane_change_count = 25

        if self.lane_change_right and self.lane_change_count > 0:
            ego = plt.Rectangle((-1, -2), 2, 4, **line_change_left_style)
            self.lane_change_count = self.lane_change_count - 1
        elif self.lane_change_left and self.lane_change_count > 0:
            ego = plt.Rectangle((-1, -2), 2, 4, **line_change_right_style)
            self.lane_change_count = self.lane_change_count - 1
        else:
            ego = plt.Rectangle((-1, -2), 2, 4, **ego_style)
            self.lane_change_right = False
            self.lane_change_left = False
        self.ax.add_patch(ego)
        plotted_objects.append(ego)

        self.plotted_objects = plotted_objects

    def remove_patches(self, ):
        # self.fig.canvas.mpl_disconnect('pick_event')
        for figure_object in self.plotted_objects:
            if isinstance(figure_object, list):
                figure_object[0].remove()
            else:
                figure_object.remove()

    def display_info(self, ego_current):
        # Get the current ego info string
        speed = ego_current['speed']
        heading = ego_current['heading']
        acc_x = ego_current['acc_x']
        acc_y = ego_current['acc_y']

        if self.cut_in_count > 0:
            cut_in = "!!!!!CUT IN!!!!!"
            self.cut_in_count = self.cut_in_count - 1
            if self.steering_ang:
                steering_ang = ego_current['steering_ang']
                msg = 'Speed: {:.2f}\nHeading: {:.2f}\nAcc_x: {:.2f}\nAcc_y: {:.2f}\nsteer_ang: {:.2f}\n\n\n{}'.format(
                    speed,
                    heading,
                    acc_x, acc_y,
                    steering_ang, cut_in)
            else:
                msg = 'Speed: {:.2f}\nHeading: {:.2f}\nAcc_x: {:.2f}\nAcc_y: {:.2f}\n\n\n{}'.format(speed, heading,
                                                                                                    acc_x, acc_y,
                                                                                                    cut_in)
        else:
            if self.steering_ang:
                steering_ang = ego_current['steering_ang']
                msg = 'Speed: {:.2f}\nHeading: {:.2f}\nAcc_x: {:.2f}\nAcc_y: {:.2f}\nsteer_ang: {:.2f}'.format(
                    speed,
                    heading,
                    acc_x, acc_y,
                    steering_ang)
            else:
                msg = 'Speed: {:.2f}\nHeading: {:.2f}\nAcc_x: {:.2f}\nAcc_y: {:.2f}'.format(speed, heading,
                                                                                            acc_x, acc_y)

        return msg

    def get_left_line(self, ego_current):

        lin_left_x_begin = ego_current['ego_line_left_begin_x']
        lin_left_x_end = ego_current['ego_line_left_end_x']
        lin_left_y_distance = ego_current['ego_line_left_distance_y']
        lin_left_curv = ego_current['ego_line_left_curv']

        lin_right_x_end = ego_current['ego_line_right_end_x']  # Make the left line and right lian has same length
        if lin_right_x_end > lin_left_x_end:
            lin_left_x_end = lin_right_x_end

        if math.isnan(lin_left_x_begin) or math.isnan(lin_left_x_end) or math.isnan(lin_left_y_distance) or math.isnan(
                lin_left_curv):
            return float('nan'), float('nan')

        lin_left_x = np.linspace(lin_left_x_begin, math.floor(lin_left_x_end), num=math.floor(lin_left_x_end))
        lin_left_y = np.ones(math.floor(lin_left_x_end)) * lin_left_y_distance

        for z in range(1, math.floor(lin_left_x_end)):
            lin_left_y[z] = lin_left_y[z] + lin_left_curv * ((z - 2) ** 2 / 2)
        return lin_left_x, lin_left_y

    def get_right_line(self, ego_current):

        lin_right_x_begin = ego_current['ego_line_right_begin_x']
        lin_right_x_end = ego_current['ego_line_right_end_x']
        lin_right_y_distance = ego_current['ego_line_right_distance_y']
        lin_right_curv = ego_current['ego_line_right_curv']

        lin_left_x_end = ego_current['ego_line_left_end_x']  # Make the left line and right lian has same length
        if lin_left_x_end > lin_right_x_end:
            lin_right_x_end = lin_left_x_end

        if math.isnan(lin_right_x_begin) or math.isnan(lin_right_x_end) or math.isnan(
                lin_right_y_distance) or math.isnan(
            lin_right_curv):
            return float('nan'), float('nan')

        lin_right_x = np.linspace(lin_right_x_begin, math.floor(lin_right_x_end), num=math.floor(lin_right_x_end))
        lin_right_y = np.ones(math.floor(lin_right_x_end)) * lin_right_y_distance

        for z in range(1, math.floor(lin_right_x_end)):
            lin_right_y[z] = lin_right_y[z] + lin_right_curv * ((z - 2) ** 2 / 2)
        return lin_right_x, lin_right_y

    @staticmethod
    def show():
        plt.show()
        plt.close()
        plt.gca().set_aspect('equal', adjustable='box')
