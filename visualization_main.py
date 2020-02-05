# Haoyu Ren
# 11.11.2019
# Visualization of data

from visualization_utils import *


if __name__ == "__main__":

    # video_path = r'..\MDM data process\video\processed_video\20191029_130130_3015.avi'
    # csv_path = r'..\MDM data process\processed_data\csv\first'

    # video_path = r'..\MDM data process\video\processed_video\20191029_133209_4315.avi'
    # csv_path = r'..\MDM data process\processed_data\csv\second'
    # file_prefix = r'0_cv'
    # video_path = r'..\ma_haoyu\processed_video\20191029_130130_3015.avi'
    # csv_path = r'..\ma_haoyu\processed_data\csv'
    file_prefix = r'0_cv'
    video_path = r'..\ma_haoyu\processed_video\20191029_133209_4315.avi'
    csv_path = r'..\ma_haoyu\processed_data\csv\second'
# =======
#     video_path = r'..\MDM data process\video\processed_video\20191029_133209_4315.avi'
#     csv_path = r'..\MDM data process\processed_data\csv\test'
#     # csv_path = r'..\MDM data process\processed_data\csv\test'
#     file_prefix = r'20_no_steering_sdf'
#     # video_path = r'..\ma_haoyu\processed_video\20191029_130130_3015.avi'
#     # csv_path = r'..\ma_haoyu\processed_data\csv'
# >>>>>>> 8a8a09d237282c56f77286390fadbd3946ebb716
    ego, dynamic_raw, static_raw = read_files(csv_path, file_prefix)
    dynamic = process_dynamic(dynamic_raw)
    static = process_static(static_raw)

    play_video = False

    try:
        if dynamic is not None and static is not None and ego is not None:
            if play_video:
                frame_count, cap = video(video_path)
                if frame_count > 0:
                    visualization_plot = VisualizationPlot(ego, dynamic, static, play_video, frame_count, cap)
                    visualization_plot.show()
                else:
                    print('Video file damaged')
            else:
                visualization_plot = VisualizationPlot(ego, dynamic, static, play_video)
                visualization_plot.show()
    except:
        print('Something went wrong when process the raw data')
        sys.exit(1)
