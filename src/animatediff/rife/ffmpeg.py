from enum import Enum
from pathlib import Path
from re import split
from typing import Annotated, Optional, Union

import ffmpeg
from ffmpeg.nodes import FilterNode, InputNode
from torch import Value


class VideoCodec(str, Enum):
    gif = "gif"
    vp9 = "vp9"
    webm = "webm"
    webp = "webp"
    h264 = "h264"
    hevc = "hevc"


def codec_extn(codec: VideoCodec):
    match codec:
        case VideoCodec.gif:
            return "gif"
        case VideoCodec.vp9:
            return "webm"
        case VideoCodec.webm:
            return "webm"
        case VideoCodec.webp:
            return "webp"
        case VideoCodec.h264:
            return "mp4"
        case VideoCodec.hevc:
            return "mp4"
        case _:
            raise ValueError(f"Unknown codec {codec}")


def clamp_gif_fps(fps: int):
    """Clamp FPS to a value that is supported by GIFs.

    GIF frame duration is measured in 1/100ths of a second, so we need to clamp the
    FPS to a value that 100 is a factor of.
    """
    # the sky is not the limit, sadly...
    if fps > 100:
        return 100

    # if 100/fps is an integer, we're good
    if 100 % fps == 0:
        return fps

    # but of course, it was never going to be that easy.
    match fps:
        case x if x > 50:
            # 50 is the highest FPS that 100 is a factor of.
            # people will ask for 60. they will get 50, and they will like it.
            return 50
        case x if x >= 30:
            return 33
        case x if x >= 24:
            return 25
        case x if x >= 20:
            return 20
        case x if x >= 15:
            # ffmpeg will pad a few frames to make this work
            return 16
        case x if x >= 12:
            return 12
        case x if x >= 10:
            # idk why anyone would request 11fps, but they're getting 10
            return 10
        case x if x >= 6:
            # also invalid but ffmpeg will pad it
            return 6
        case 4:
            return 4  # FINE, I GUESS
        case _:
            return 1  # I don't know why you would want this, but here you go


class FfmpegEncoder:
    def __init__(
        self,
        frames_dir: Path,
        out_file: Path,
        codec: VideoCodec,
        in_fps: int = 60,
        out_fps: int = 60,
        lossless: bool = False,
        param={},
    ):
        self.frames_dir = frames_dir
        self.out_file = out_file
        self.codec = codec
        self.in_fps = in_fps
        self.out_fps = out_fps
        self.lossless = lossless
        self.param = param

        self.input: Optional[InputNode] = None

    def encode(self) -> tuple:
        self.input: InputNode = ffmpeg.input(
            str(self.frames_dir.resolve().joinpath("%08d.png")), framerate=self.in_fps, loglevel="warning"
        ).filter("fps", fps=self.in_fps)
        match self.codec:
            case VideoCodec.gif:
                return self._encode_gif()
            case VideoCodec.webm:
                return self._encode_webm()
            case VideoCodec.webp:
                return self._encode_webp()
            case VideoCodec.h264:
                return self._encode_h264()
            case VideoCodec.hevc:
                return self._encode_hevc()
            case _:
                raise ValueError(f"Unknown codec {self.codec}")

    @property
    def _out_file(self) -> Path:
        return str(self.out_file.resolve())

    @staticmethod
    def _interpolate(stream, out_fps: int) -> FilterNode:
        return stream.filter(
            "minterpolate", fps=out_fps, mi_mode="mci", mc_mode="aobmc", me_mode="bidir", vsbmc=1
        )

    def _encode_gif(self) -> tuple:
        stream: FilterNode = self.input

        # Output FPS must be divisible by 100 for GIFs, so we clamp it
        out_fps = clamp_gif_fps(self.out_fps)
        if self.in_fps != out_fps:
            stream = self._interpolate(stream, out_fps)

        # split into two streams for palettegen and paletteuse
        split_stream = stream.split()

        # generate the palette, then use it to encode the GIF
        palette = split_stream[0].filter("palettegen")
        stream = ffmpeg.filter([split_stream[1], palette], "paletteuse").output(
            self._out_file, vcodec="gif", loop=0
        )
        return stream.run()

    def _encode_webm(self) -> tuple:
        stream: FilterNode = self.input
        if self.in_fps != self.out_fps:
            stream = self._interpolate(stream, self.out_fps)
        param = {
            "pix_fmt":"yuv420p",
            "vcodec":"libvpx-vp9",
            "video_bitrate":0,
            "crf":24,
        }
        param.update(**self.param)
        stream = stream.output(
            self._out_file, **param
        )
        return stream.run()

    def _encode_webp(self) -> tuple:
        stream: FilterNode = self.input
        if self.in_fps != self.out_fps:
            stream = self._interpolate(stream, self.out_fps)

        if self.lossless:
            param = {
                "pix_fmt":"bgra",
                "vcodec":"libwebp_anim",
                "lossless":1,
                "compression_level":5,
                "qscale":75,
                "loop":0,
            }
            param.update(**self.param)
            stream = stream.output(
                self._out_file,
                **param
            )
        else:
            param = {
                "pix_fmt":"yuv420p",
                "vcodec":"libwebp_anim",
                "lossless":0,
                "compression_level":5,
                "qscale":90,
                "loop":0,
            }
            param.update(**self.param)
            stream = stream.output(
                self._out_file,
                **param
            )
        return stream.run()

    def _encode_h264(self) -> tuple:
        stream: FilterNode = self.input
        if self.in_fps != self.out_fps:
            stream = self._interpolate(stream, self.out_fps)

        param = {
            "pix_fmt":"yuv420p",
            "vcodec":"libx264",
            "crf":21,
            "tune":"animation",
        }
        param.update(**self.param)

        stream = stream.output(
            self._out_file, **param
        )
        return stream.run()

    def _encode_hevc(self) -> tuple:
        stream: FilterNode = self.input
        if self.in_fps != self.out_fps:
            stream = self._interpolate(stream, self.out_fps)

        param = {
            "pix_fmt":"yuv420p",
            "vcodec":"libx264",
            "crf":21,
            "tune":"animation",
        }
        param.update(**self.param)

        stream = stream.output(self._out_file, **param)
        return stream.run()
