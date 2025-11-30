import cv2

def video_irora_keszit(vid, kimenet_utvonal, fps=None):
    if fps is None:
        fps = vid.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

    szeles = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    magas = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    codec = cv2.VideoWriter_fourcc(*"mp4v")
    iro = cv2.VideoWriter(kimenet_utvonal, codec, fps, (szeles, magas))
    return iro
