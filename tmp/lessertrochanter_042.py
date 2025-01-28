# © Copyright 2021 Naitive Technologies Ltd
"""
Synthetisis of Lesser Trochanter x-rays.
"""

import math
from typing import Dict, List, Tuple, Union, Optional
from random import randint, uniform, choice, getrandbits

import numpy as np
import cv2  # type:ignore

from .. import xraysynthesis
from ..textures import upsampledgaussiantexture, xraynoisetexture
from ..utils import beziercurve
from ..utils.imagemanipulation import (  # type:ignore
    draw_straight_lines,
    link_curves,
    fill_poly_with_texture,
    generate_attenuation_array,
    combine_with_overlay,
    add_text_to_image,
    normalise_pixel_intensity,
)
from coreutils.imageutils.pixelmanipulation import mask_check  # type:ignore
from coreutils.imageutils.imagedisplay import (  # type:ignore
    display_images_actual_size_grid,
)  # type:ignore

BONE_POINT = {
    "a3": np.array([[10, -5]]),
    "a2": np.array([[128, 320]]),
    "a1": np.array([[128, 640 + 10]]),
}

# Points defining bone variation [see README.md diagram]
BONE_POINTS = [BONE_POINT]
# Corresponding (row, col) image scales of these points.
ORIGINAL_POINT_SCALES = [(640.0, 640.0)]


def add_linear_artefacts(image: np.ndarray) -> np.ndarray:
    """Add dark linear artefacts to synthetic image.

    Parameters
    ----------
    image: np.ndarray
        Synthetic image to add artefacts to.

    Returns
    -------
    np.ndarray
        Synthetic image with artefacts added.

    """
    num = randint(0, 10)
    if num > 6:
        s = randint(20, 150)
        e1 = np.asarray([-5, randint(75, 200)])
        e3 = np.asarray([260, e1[1] - randint(0, 10) - s])
        e2 = np.asarray([-5, e1[1] - randint(10, 25)])
        e4 = np.asarray([260, e2[1] - randint(5, 30) - s])
        artefact_points = np.asarray([e1, e2, e4, e3])
        artefact = cv2.fillPoly(
            np.zeros(image.shape),
            [
                artefact_points.astype(np.int32),
            ],
            -1 * uniform(0.5, 0.9) * image.max(),
        )
        image = combine_with_overlay(image, artefact, 1 / 4)
    num = randint(0, 10)
    if num > 8:
        s = randint(20, 150)
        e1 = np.asarray([-5, randint(75, 200)])
        e3 = np.asarray([260, e1[1] - randint(0, 10) - s])
        e2 = np.asarray([-5, -5])
        e4 = np.asarray([260, -5])
        artefact_points = np.asarray([e1, e2, e4, e3])
        artefact = cv2.fillPoly(
            np.zeros(image.shape),
            [artefact_points.astype(np.int32)],
            uniform(0.6, 1.1) * image.max(),
        )
        image = combine_with_overlay(image, artefact, 1 / 4)

    return image


def shift(array, d, r):
    """?"""
    t = np.array((range(len(array))))
    z = t * 0
    t = np.asarray([math.floor(d * (1 - math.exp(-i / r))) for i in t])
    return np.stack((t, z), axis=1)


def create_c(p, val):
    """?"""
    c = []
    for i in range(int((len(p) - 1) / 2)):
        c.append((i, val))
    return c


class LesserTrochanterXRaySynthesis(xraysynthesis.XRaySynthesis):
    """X-Ray image synthesis for the Lesser Trochanter."""

    def __init__(
        self,
        zoom=True,
        simple=True,
        bone_points: List[Dict[str, np.ndarray]] = BONE_POINTS,
        original_point_scales: List[Tuple[float, float]] = ORIGINAL_POINT_SCALES,
        view_limits: Tuple[int, int, int, int] = (40, 65, 0, 25),
        scale_limits: Tuple[float, float] = (0.8, 1.4),
        angle_limits: Tuple[float, float] = (-20, 10),
        label_side="LEFT",
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        zoom: bool
            If True, the image produced is zoomed in to the femoral head.
        simple: bool
            If True, the pelvis is drawn as a simpler shape.
        bone_points: List[Dict[str, np.ndarray]
            The list of dictionaries of points definining the outline of hip sections.
        original_point_scales: List[Tuple[float, float]]
            The original dimensions of the images that the points dictionaries in hip_points
            are based on. Must have the same length as hip_points.
        view_limits: Tuple[int, int, int, int]
            The x and y limits of the hip image's random translation vector.
        scale_limits: Tuple[float, float]
            The upper and lower bounds of the scale of the x-ray image.
        angle_limits: Tuple[float, float]
            The upper and lower bounds of the angle of the x-ray image.
        label_side: str
            The side of the hip to be labeled, and cropped into if zoom is true.

        """
        super().__init__(**kwargs)

        self.zoom = zoom
        self.simple = simple

        self.bone_points = bone_points
        self.original_point_scales = original_point_scales
        self.view_limits = view_limits
        self.scale_limits = scale_limits
        self.angle_limits = angle_limits
        self.label_side = label_side

    def scale_points(
        self,
        points: Dict[str, np.ndarray],
        original_scale: Tuple[float, float],
        new_scale: Tuple[float, float],
    ) -> Dict[str, np.ndarray]:
        """Scales dictionary of points based on their original scale.

        Parameters
        ----------
        points: Dict[str, np.ndarray]
            The dictionary of points to be modified.
        original_scale: Tuple[float, float]
            The original dimensions of the image from which the points are defined.
        new_scale: Tuple[float, float]
            The new dimensions to which the points should be scaled.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary of the scaled points.

        """
        for k, v in points.items():
            points[k] = np.array(
                [
                    v[:, 0] * (new_scale[0] / original_scale[0]),
                    v[:, 1] * (new_scale[1] / original_scale[1]),
                ]
            ).T.astype(np.int32)
        return points

    def generate_region_of_interest_and_labels(
        self, roi_magnitude: float
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]],
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    ]:
        """Generates synthetic x-ray image and its labels.

        Parameters
        ----------
        roi_magnitude: float
            The pixel intensity value used to draw the region of interest image.

        Returns
        -------
        Union[
            Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]],
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            ]
            (numpy array, numpy array, numpy array), pixel data of the synthetic x-ray image,
            the label image respectively.

        """
        image_size = self.image_size

        # Choose marker points
        scaled_points = self.scale_points(
            self.bone_points[0].copy(),
            self.original_point_scales[0],
            (image_size, image_size),
        )

        # DEFINE DRAWING CONSTANTS
        roi = roi_magnitude * uniform(1, 1)
        trabecular_texture = upsampledgaussiantexture.UpsampledGaussianTexture(
            64 / 80, 8 / 80
        )
        cbone = uniform(0.8, 1)  # cortical bone intensity
        cflesh = max(0, uniform(-0.4, 0.3))
        min_trab = uniform(0.35, 0.5)  # minimum trabeular intensity
        max_trab = uniform(0.6, 0.9)  # maximum trabelular intensity
        end_inner, end_inner_extra = randint(19, 22), randint(
            1, 2
        )  # random variation where inner-cortical finishes
        end_outter, end_outter_extra = randint(19, 22), randint(
            0, 1
        )  # random variation where outer-cortical finishes

        s = randint(20, 140)  # overall shift
        w = randint(-40, -8)  # width shif
        t = randint(3, 6)  # thickness of hidden lt
        m = uniform(1, 1.5)

        # random variation of a/d points
        a3_x = randint(0, 50)
        a2_x = randint(-10, 20)
        a1_x = a2_x + randint(-30, 0)

        a3_y = randint(0, 0)
        a2_y = randint(-80, 0)
        a1_y = randint(32, 80)

        a1 = scaled_points["a1"] + [[s, 0]] + [[a1_x, a1_y]]
        a2 = scaled_points["a2"] + [[s // 2, 0]] + [[a2_x, a2_y]]
        a3 = scaled_points["a3"] + [[0, 0]] + [[a3_x, a3_y]]

        points = {}
        points["inner_cortical_a_bottom"] = np.concatenate((a1, a2, a3), axis=0)

        # ==================== GENERATE ROI IMAGE =====================
        bone_image = np.ones((image_size, image_size), dtype=np.float32) * uniform(
            0, 0.05
        )

        # CORTICAL LINES
        bone_image, a_points = beziercurve.draw_bezier_curves(
            bone_image,
            points["inner_cortical_a_bottom"],
            color=1,
            thickness=1,
            curvatures=[(0, 0.5)],
            return_all_points=True,
        )

        whole_points = link_curves(
            a_points, [np.asarray([520, 520]), np.asarray([520, -5])]
        )
        a_array = np.asarray(a_points)
        # c_array = np.asarray(d_points[::-1][0:end_outter])
        # c_array_shifted = (c_array - np.flipud(shift(c_array,randint(36,48) + (32+w)//2,randint(26,36)))).tolist()
        b_array = (
            a_array
            + [randint(5, 30), 0]
            + np.flipud(
                shift(a_array, randint(10, 40) + (32 + w) // 2, randint(10, 18))
            )
        ).tolist()
        b_points = [np.asarray(x) for x in b_array]

        points["inner_cortical_b_bottom"] = b_array
        half_points = link_curves(
            b_points, [np.asarray([520, 520]), np.asarray([520, -5])]
        )

        # FILL TRABECULAR WITH TEXTURE
        bone_image = fill_poly_with_texture(
            bone_image,
            whole_points.astype(np.int32),
            trabecular_texture.generate(bone_image.shape),
            max_color=roi * max_trab,
            min_color=roi * min_trab,
        )

        inner_cortical_points = link_curves(a_array, b_array)
        # outter_cortical_points = link_curves(c_array, c_array_shifted)

        # FILL CORTICAL WITH TEXTURE
        bone_image = fill_poly_with_texture(
            bone_image,
            inner_cortical_points.astype(np.int32),
            np.ones(bone_image.shape),
            max_color=m * roi * cbone,
            min_color=roi * cbone,
        )
        # bone_image = fill_poly_with_texture(bone_image, np.int32(outter_cortical_points), np.ones(bone_image.shape), max_color=roi*cbone, min_color=roi*cbone)

        if randint(0, 10) < 9:
            hidden = False
        else:
            hidden = True

        if not hidden:
            # Lesser trochanter protruding
            len_lt = randint(4, 7)
            start_lt = randint(4, 10)
            col_lt = roi
            inner_lt = np.asarray(
                [
                    b_points[start_lt],  # curve[0],
                    np.asarray(
                        [
                            min(b_points[start_lt][0], b_points[start_lt + len_lt][0])
                            - randint(5, 75),
                            randint(-15, 40)
                            + (b_points[start_lt][1] + b_points[start_lt + len_lt][1])
                            / 2,
                        ]
                    ),
                    b_points[start_lt + len_lt],  # curve[-1]
                ]
            )
            _, troch_points = beziercurve.draw_bezier_curves(
                np.zeros(bone_image.shape),
                np.array(inner_lt.reshape(3, 2)).astype(np.int32),
                color=0,
                thickness=1,
                curvatures=[(0, 0.7)],
                return_all_points=True,
            )
            troch_points += [np.asarray([520, 256])]  # curve
            overlay = np.zeros(bone_image.shape)
            overlay = fill_poly_with_texture(
                overlay,
                np.array(troch_points).astype(np.int32),
                trabecular_texture.generate(bone_image.shape),
                max_color=roi * max_trab * 0.8,
                min_color=roi * min_trab * 0.8,
            )
            overlay = cv2.fillPoly(
                overlay,
                [
                    np.array(half_points).astype(np.int32),
                ],
                0,
            )
            bone_image = combine_with_overlay(bone_image, overlay, 1)
        elif hidden:
            pass
            # Lesser trochanter hidden behind cortical
            len_lt = randint(2, 4)
            start_lt = randint(3, 11)
            col_lt = roi
            inner_lt = np.asarray(
                [
                    b_points[start_lt],  # curve[0],
                    np.asarray(
                        [
                            min(b_points[start_lt][0], b_points[start_lt + len_lt][0])
                            - randint(0, 15),
                            randint(-20, 20)
                            + (b_points[start_lt][1] + b_points[start_lt + len_lt][1])
                            / 2,
                        ]
                    ),
                    b_points[start_lt + len_lt],  # curve[-1]
                ]
            )
            inner_lt += [40, 0]
            bone_image, troch_points = beziercurve.draw_bezier_curves(
                bone_image,
                np.array(inner_lt.reshape(3, 2)).astype(np.int32),
                color=m * roi,
                thickness=t,
                curvatures=[(0, 0.7)],
                return_all_points=True,
            )

        # Linear Artefacts`
        if bool(getrandbits(1)):
            bone_image = add_linear_artefacts(bone_image)

        vectors = [(-1, 1), (-1, -1), (-1, 0), (1, -1), (1, 1), (1, 0), (0, 1), (0, -1)]
        attenuation_array = generate_attenuation_array(
            bone_image.shape, choice(vectors), 0, uniform(0.5, 1)
        )
        bone_image *= attenuation_array

        # ==================== GENERATE LABEL IMAGE1 =====================
        label_image1 = np.zeros((image_size, image_size), dtype=np.float32)
        label_image1 = cv2.fillPoly(
            label_image1,
            [
                whole_points.astype(np.int32),
            ],
            1,
        )
        label_image1 = draw_straight_lines(
            label_image1, a_points[0 : end_inner + end_inner_extra], 1, 2
        )
        # label_image1 = draw_straight_lines(label_image1, d_points[::-1][0:end_outter+end_outter_extra],1,2)

        # ==================== GENERATE LABEL IMAGE2 =====================
        label_image2 = np.zeros((image_size, image_size), dtype=np.float32)
        if not hidden:
            label_image2 = cv2.fillPoly(
                label_image2,
                [
                    np.array(troch_points).astype(np.int32),
                ],
                1,
            )
            label_image2 = cv2.fillPoly(
                label_image2,
                [
                    half_points.astype(np.int32),
                ],
                0,
            )
        elif hidden:
            label_image2, troch_points = beziercurve.draw_bezier_curves(
                label_image2,
                inner_lt.reshape(3, 2).astype(np.int32),
                color=1,
                thickness=t,
                curvatures=[(0, 0.7)],
                return_all_points=True,
            )
            pass

        # RESIZE
        bone_image = cv2.resize(bone_image, (self.image_size, self.image_size))
        label_image1 = cv2.resize(label_image1, (self.image_size, self.image_size))
        label_image2 = cv2.resize(label_image2, (self.image_size, self.image_size))

        # SET MASKS TO 0 or 1 after resize
        label_image1 = mask_check(label_image1)
        label_image2 = mask_check(label_image2)

        return (bone_image, label_image1, label_image2)

    def create_image_with_xray_noise_and_label(
        self,
        return_no_noise_image: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Creates full synthetic x-ray image with added noise layer.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            - 2D array of synthetic x-ray image,
            - Corresponding label image of ?
            - Corresponding label image of ?

        Note
        ----
        Variables of parent class must be present.

        """
        xray_noise_texture = xraynoisetexture.XRayNoiseTexture(
            self.noise_background_pixel_intensity,
            uniform(*self.image_generation_gaussian_variance_range),
            uniform(*self.noise_generation_gaussian_variance_range),
        )
        synthetic_image = xray_noise_texture.generate(
            (self.image_size, self.image_size)
        )
        # Use a random multiple of the standard deviation as the magnitude of the pixel intensity increase in the roi
        roi_magnitude = uniform(
            self.roi_magnitude_range[0], self.roi_magnitude_range[1]
        ) * np.std(synthetic_image)
        (
            region_of_interest_image,
            label_image1,
            label_image2,
        ) = self.generate_region_of_interest_and_labels(roi_magnitude)
        if self.place_random_text:
            for i in range(np.random.randint(7)):
                add_text_to_image(region_of_interest_image, roi_magnitude)

        if self.max_blur > 1:
            blur_amount = np.random.randint(2, self.max_blur)
            region_of_interest_image = cv2.blur(
                region_of_interest_image, (blur_amount, blur_amount)
            )
        synthetic_image = (
            uniform(0.5, 1.5) * synthetic_image
        ) + region_of_interest_image
        attenuation_array = generate_attenuation_array(
            synthetic_image.shape, (-1, 0), uniform(0.3, 0.8), 1
        )
        synthetic_image *= attenuation_array
        synthetic_image = normalise_pixel_intensity(synthetic_image)

        return (synthetic_image, label_image1, label_image2)

    def create_training_data(
        self,
        dataset_size: int = 400,
        training_fraction: float = 0.5,
        real_data: List[np.ndarray] = [],
        real_labels: List[np.ndarray] = [],
        real_training_fraction: float = 1.0,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create training and validation synthetic datasets.

        Parameters
        ----------
        dataset_size: int
            The total number of synthetic images to produce.
        real_data: [numpy array, ...]
            An array of pixel data of real x-ray images. These are added to the training set.
        real_labels: [numpy array, ...]
            The corresponding labels defining the correct segmentation of the given real_data.

        Note
        ----
        Variables of parent class must be present.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Arrays of the training data, training labels, test data, and test labels respectively.

        """
        data: List[np.ndarray] = []
        labels: List[np.ndarray] = []

        for _ in range(dataset_size):
            (
                synthetic_image,
                _,
                label_image2,
            ) = self.create_image_with_xray_noise_and_label()

            data.append(synthetic_image)
            labels.append(label_image2)

        # Divide into training and test set, add real data to training set
        training_data = np.array(data[0 : int(len(data) / 2)] + real_data)
        training_labels = np.array(labels[0 : int(len(data) / 2)] + real_labels)

        test_data = np.array(data[int(len(data) / 2) :])
        test_labels = np.array(labels[int(len(data) / 2) :])

        # Test plots
        display_images_actual_size_grid(
            [
                test_data[0].reshape((self.image_size, self.image_size)),
                test_labels[0].reshape((self.image_size, self.image_size)),
            ],
            cmaps=["gray", "inferno"],
            grid_width=2,
        )

        return training_data, training_labels, test_data, test_labels

    def generate_check(self):
        """Generate QC images."""
        images = self.generate_region_of_interest_and_labels(17)
        _ = display_images_actual_size_grid(
            [images[0], images[1], images[2]],
            grid_width=3,
            cmaps=["gray", "inferno", "inferno"],
        )
        return images

    def show_examples(self, image_count: int = 40) -> None:
        """Displays multiple examples of final image synthesis output.

        Parameters
        ----------
        image_count: int
            The number of example images displayed.

        """
        images = []

        # Create example images
        for i in range(image_count):
            images.append(self.create_image_with_xray_noise_and_label()[0])

        display_images_actual_size_grid(images, grid_width=5)

    def show_full_example(self) -> None:
        """Displays single example image with full noise, no noise, and label versions."""
        xray_noise_texture = xraynoisetexture.XRayNoiseTexture(
            self.noise_background_pixel_intensity,
            uniform(*self.image_generation_gaussian_variance_range),
            uniform(*self.noise_generation_gaussian_variance_range),
        )
        synthetic_image = xray_noise_texture.generate(
            (self.image_size, self.image_size)
        )
        # Use a random multiple of the standard deviation as the magnitude of the pixel intensity increase in the roi
        roi_magnitude = uniform(
            self.roi_magnitude_range[0], self.roi_magnitude_range[1]
        ) * np.std(synthetic_image)
        (
            region_of_interest_image,
            label_image,
            label_image2,
        ) = self.generate_region_of_interest_and_labels(roi_magnitude)
        if self.place_random_text:
            for i in range(np.random.randint(7)):
                add_text_to_image(region_of_interest_image, roi_magnitude)

        # Apply blur to synthetic x-ray image
        if self.max_blur > 1:
            blur_amount = np.random.randint(2, self.max_blur)
            region_of_interest_image = cv2.blur(
                region_of_interest_image, (blur_amount, blur_amount)
            )
        if self.draw_noise:
            synthetic_image += region_of_interest_image
        else:
            synthetic_image = region_of_interest_image
        synthetic_image = normalise_pixel_intensity(synthetic_image)

        display_images_actual_size_grid(
            [synthetic_image, region_of_interest_image, label_image2],
            cmaps=["gray", "gray", "inferno"],
        )
