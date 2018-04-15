
from moviepy.editor import VideoFileClip

test_video_output = "./project_video_hard_section_0.mp4"
test_clip = VideoFileClip("./project_video.mp4").subclip(00.00, 06.00)
test_clip.write_videofile(test_video_output, audio=False)

test_video_output = "./project_video_hard_section_1.mp4"
test_clip = VideoFileClip("./project_video.mp4").subclip(18.00, 29.00)
test_clip.write_videofile(test_video_output, audio=False)

test_video_output = "./project_video_hard_section_2.mp4"
test_clip = VideoFileClip("./project_video.mp4").subclip(36.00, 46.00)
test_clip.write_videofile(test_video_output, audio=False)

test_video_output = "./project_video_hard_section_3.mp4"
test_clip = VideoFileClip("./project_video.mp4").subclip(22.00, 29.00)
test_clip.write_videofile(test_video_output, audio=False)