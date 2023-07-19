# GetVisitorFlow
The repository contains a object recognition software for detecting pedestrians, bicycles, and cars. Beside detection, the software counts the movement of the objects distinguishing the direction (left to right, right to left). The software was developed in the course of the project [DIH-West](https://dih-west.at/) together with students from the [Kufstein University of Applied Science](https://www.fh-kufstein.ac.at/) in the Master study progra [Smart Products & Solutions](https://www.fh-kufstein.ac.at/studieren/master/smart-products-solutions-bb).
## Aim of the project
The aim of the project was to develope a stamd-alone system for measuring the visitor flow in a touristic region. The systems should be capable to distinguish between pedestrians, bicycles, and cars. Beside detection, the system should also count and store the number of object passings in a direction-sensitive approach. The system should be tested in the region of Leogang (Schwarzleotal) together with the TVB Leogang-Saalfelden and Bergbahn Leogang.
## Hardware
A Raspberry Pi 4 Model B with 8 GB RAM was used. For image acquisition, a RPI WCAM with 5MP was used. The hardware components were placed inside a birdhouse. Energy supply was accomplisehd by using a Mobisun powerbank with 41580 mAh. To protect the powerbank, a custommade housing of PMMA was fabricated and attached to the birdhouse.
## Software
The operating system was Raspberry Pi OS (Raspbian). The primary library for object detection was TensorFlow Lite. The core software is based on the work from [Evan Juras](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/tree/master) and was adapted. The following changes were implemented:
- Implemention of detection lines for object counting
- Visualisation of detection line for debugging
- Saving of counted objects without saving an image or the stream
