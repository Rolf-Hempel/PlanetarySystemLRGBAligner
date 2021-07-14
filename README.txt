PlanetarySystemLRGBAligner

Python application for the automatic creation of LRGB composites from B/W and color images of planetary system objects

Imaging of planetary system objects, such as the Moon or planets, at high magnification suffers from the degrading effect of the Earth's atmosphere. Since this effect grows with decreasing wavelength, images are sharpest if taken through red or infrared filters. Color images, on the other hand, suffer from strong seeing effects at shorter wavelengths.

By combining a sharp B/W image taken in the near infrared with a color image with much less resolution, the LRGB technique has the potential to create a color image with the full resolution of the near-infrared image. Technically, the B/W image is assigned to the "luminance" channel of the combined image, while the color image's RGB channels only provide the color information.

For this to work properly, both images first have to be registered pixel-wise with high precision. For extended objects such as the moon, this requires more than a simple affine mapping. It is, therefore, impossible to do the registration manually in an image processing program.

PlanetarySystemLRGBAligner does this image registration automatically using a two phase approach: First, it computes a rigid homography mapping to match the color image approximately with the B/W image. In a second phase, an "optical flow" algorithm adjusts the registration of both images at every pixel location.

As an add-on, when the registration is completed, PlanetarySystemLRGBAligner offers to combine both images into an LRGB composite.