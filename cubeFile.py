import os

import numpy as np
import numpy.typing as npt

import logging
import logging.config

from typing import Self, TextIO, Tuple

logging.config.fileConfig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging.conf")
)
loggerErr = logging.getLogger("stderr")

loggerOut = logging.getLogger("stdout")
loggerDebug = logging.getLogger("debug")


class CubeFile:
    """Class Cube File"""

    elementNumToString = {
        0: "udf",
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "Ne",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        18: "Ar",
        19: "K",
        20: "Ca",
        21: "Sc",
        22: "Ti",
        23: "V",
        24: "Cr",
        25: "Mn",
        26: "Fe",
        27: "Co",
        28: "Ni",
        29: "Cu",
        30: "Zn",
        31: "Ga",
        32: "Ge",
        33: "As",
        34: "Se",
        35: "Br",
        36: "Kr",
        37: "Rb",
        38: "Sr",
        39: "Y",
        40: "Zr",
        41: "Nb",
        42: "Mo",
        43: "Tc",
        44: "Ru",
        45: "Rh",
        46: "Pd",
        47: "Ag",
        48: "Cd",
        49: "In",
        50: "Sn",
        51: "Sb",
        52: "Te",
        53: "I",
        54: "Xe",
        55: "Cs",
        56: "Ba",
        57: "La",
        58: "Ce",
        59: "Pr",
        60: "Nd",
        61: "Pm",
        62: "Sm",
        63: "Eu",
        64: "Gd",
        65: "Tb",
        66: "Dy",
        67: "Ho",
        68: "Er",
        69: "Tm",
        70: "Yb",
        71: "Lu",
        72: "Hf",
        73: "Ta",
        74: "W",
        75: "Re",
        76: "Os",
        77: "Ir",
        78: "Pt",
        79: "Au",
        80: "Hg",
        81: "Tl",
        82: "Pb",
        83: "Bi",
        84: "Po",
        85: "At",
        86: "Rn",
        87: "Fr",
        88: "Ra",
        89: "Ac",
        90: "Th",
        91: "Pa",
        92: "U",
        93: "Np",
        94: "Pu",
        95: "Am",
        96: "Cm",
    }

    def __init__(self, cubeFilePath: str) -> None:
        """Cube File init"""
        self.cubeFilePath = cubeFilePath

    def __enter__(self) -> Self:
        """Cube File context manager enter"""
        try:
            self.cubeFile = open(self.cubeFilePath, "r")
            self._readCubeFile(self.cubeFile)
            self.cubeFile.close
        except IOError as e:
            loggerErr.exception(f"CubeFile: {self.cubeFilePath} could not be read.")
        loggerOut.info(f"Finished reading cube file {self.cubeFilePath}")

        return self

    def __exit__(self, type, value, traceback):
        """Cube File context manager exit"""
        del self

    def _readCubeFile(self, f: TextIO) -> None:
        """readCubeFile
        Read the Cube File, store information about the system located in the header of the cube file in np.arrays
        and store the data of the cube file in an np.array

        The following class attributes are created:
        - numberOfAtoms
        - numberOfVoxelsX, numberOfVoxelsY, numberOfVoxelsZ
        - voxelVectorX, voxelVectorY, voxelVectorZ
        - nameOfAtoms
        - chargeOfAtoms
        - coordinatesOfAtoms
        - simulationBoxSize
        - data

        params:
                None
        returns:
                None
        """

        # Line 1 and two are comments and are ignored
        f.readline()
        f.readline()

        # Read the file header
        self.numberOfAtoms, self.origin = self._readHeaderLineShort(f)
        self.numberOfVoxelsX, self.voxelVectorX = self._readHeaderLineShort(f)
        self.numberOfVoxelsY, self.voxelVectorY = self._readHeaderLineShort(f)
        self.numberOfVoxelsZ, self.voxelVectorZ = self._readHeaderLineShort(f)

        self.nameOfAtoms = np.empty((self.numberOfAtoms), str)
        self.chargeOfAtoms = np.empty((self.numberOfAtoms), int)
        self.coordinatesOfAtoms = np.empty((self.numberOfAtoms, 3), float)
        for i in range(self.numberOfAtoms):
            element, charge, coordinates = self._readHeaderLineLong(f)
            self.nameOfAtoms[i] = self.elementNumToString[element]
            self.chargeOfAtoms[i] = charge
            self.coordinatesOfAtoms[i] = np.array([coordinates])

        # Determine the units of the cube file from file header
        if (
            self.numberOfVoxelsX < 0
            or self.numberOfVoxelsX < 0
            or self.numberOfVoxelsX < 0
        ):
            self.unit = "Angstrom"
            self.numberOfVoxelsX = abs(self.numberOfVoxelsX)
            self.numberOfVoxelsY = abs(self.numberOfVoxelsY)
            self.numberOfVoxelsZ = abs(self.numberOfVoxelsZ)
        else:
            self.unit = "Bohr"

        # Determine voxel size from file header
        self.voxelSizeX = self.voxelVectorX[0]
        self.voxelSizeY = self.voxelVectorY[1]
        self.voxelSizeZ = self.voxelVectorZ[2]

        # Determine simulation box size from file header
        self.simulationBoxSize = np.array(
            [
                (self.numberOfVoxelsX - 1) * self.voxelSizeX,
                (self.numberOfVoxelsY - 1) * self.voxelSizeY,
                (self.numberOfVoxelsZ - 1) * self.voxelSizeZ,
            ]
        )

        # read the data of the cube file
        self.data = np.zeros(
            (self.numberOfVoxelsX * self.numberOfVoxelsY * self.numberOfVoxelsZ)
        )
        idx = 0
        for line in f:
            for value in line.split():

                # This happens when the value is smaller than E99. For E100 is not enogh space and the E is ommit (e. g. 0.806033-100)
                if "E" not in value:
                    value = 0

                try:
                    self.data[idx] = float(value)
                except ValueError:
                    loggerErr.exception("Failed to convert string to float")

                idx += 1

        self.data = np.reshape(
            self.data,
            (self.numberOfVoxelsX, self.numberOfVoxelsY, self.numberOfVoxelsZ),
        )

    def _readHeaderLineShort(self, file: TextIO) -> Tuple[int, np.ndarray]:
        """Read a single line of the header of the cube file and return the leading int and
            the float values that follow it.
            Short means it has only one leading int and three float values
        params:
                file (instance of the file)
        return:
                leadingInt, listOfFloats ((int, np.array([])))
        """
        lineSplit = file.readline().split()

        leadingInt = int(lineSplit[0])
        listOfFloats = np.array(list(map(float, lineSplit[1:])))

        return leadingInt, listOfFloats

    def _readHeaderLineLong(self, file: TextIO) -> Tuple[int, float, np.ndarray]:
        """Read a single line of the header of the cube file and return the leading int, float value
            the  rest of float values that follow it as a list.
            Long means it has only one leading int and four float values
        params:
                file (instance of the file)
        return:
                leadingInt, firstFloat, listOfFloats ((int, float, np.array([])))
        """
        leadingInt, listOfFloats = self._readHeaderLineShort(file)
        return leadingInt, listOfFloats[0], listOfFloats[1:]


    def integrateEntireCubeData(self) -> float:
        ''' Integratet (sum) all the data of the cube file
            This is the number of electrons fi the cube file contains electron denstiy
        params: 
                None
        return: 
                numberOfElectrons (float): The result of the integration
        '''
        volume = self.voxelSizeX * self.voxelSizeY * self.voxelSizeZ
        electronDensity = np.sum(self.data)
        numberOfElectrons = volume * electronDensity

        return numberOfElectrons
    

    def integrateCubicRegion(self, startVoxels: npt.ArrayLike, endVoxels: npt.ArrayLike) -> float:
        ''' Integratet (sum) a cubic region of the cube file.
            The region is defined by start Voxesl in 3 dimension e.g. [0, 0, 15] and
            end Voxesl in 3 dimension e.g. [12, 12, 20]
            This would sum up the region (0->12, 0->12, 15->20)
        params: 
                startVoxels (np.array([], dtype=int)): The "coordinates" Voxels of the start of the region
                endVoxels (np.array([], dtype=int)): The "coordinates" Voxels of the end of the region
        return: 
                numberOfElectrons (float): The result of the integration
        '''
        assert len(startVoxels) == len(endVoxels) == 3

        assert startVoxels[0] < endVoxels[0]
        assert startVoxels[1] < endVoxels[1]
        assert startVoxels[2] < endVoxels[2]

        assert endVoxels[0] <= self.numberOfVoxelsX
        assert endVoxels[1] <= self.numberOfVoxelsY
        assert endVoxels[2] <= self.numberOfVoxelsZ

        electronDensity = np.sum(self.data[startVoxels[0]:endVoxels[0],
                                           startVoxels[1]:endVoxels[1],
                                           startVoxels[2]:endVoxels[2],
                                           ])

        volume = self.voxelSizeX * self.voxelSizeY * self.voxelSizeZ

        numberOfElectrons = volume * electronDensity

        return numberOfElectrons
    

    def integrateCubeDataAroundAtoms(self, indices: npt.ArrayLike, integrationRadius:float=7.0) -> float:
        ''' Integratet (sum) a spherical region around atoms.
            The atoms are defined by their index.
            The spherical region is defined by an radius and the coordinates of the atoms
        params: 
                indices (np.array([])): The indices of the atoms
                integrationRadius (float): The radius of the spheres
        return: 
                numberOfElectrons (float): The result of the integration
        '''
        coordinatesOfReferenceAtoms = self.coordinatesOfAtoms[indices, :]
        coordinatesOfReferenceAtoms = coordinatesOfReferenceAtoms + self.simulationBoxSize/2

        numberOfElectrons = self.integrateSpheres(coordinatesOfReferenceAtoms, integrationRadius)
    
        return numberOfElectrons


    def integrateSpheres(self, referenceCoordinates: npt.ArrayLike, integrationRadius: float) -> float:
        ''' Integratet (sum) a spherical region
            The spherical region is defined by an referenceCoordinates and a radius.
        params: 
                referenceCoordinates (np.array([])): the xyz coordinates of the sphere
                integrationRadius (float): The radius of the spheres
        return: 
                numberOfElectrons (float): The result of the integration
        '''
        if not isinstance(referenceCoordinates, np.ndarray):
            referenceCoordinates = np.array(referenceCoordinates)

        # make sure refCoordinatesBohr has two dimensions
        if len(referenceCoordinates.shape) == 1:
            referenceCoordinates = np.array([referenceCoordinates])

        # create a mask to lay over the cube data
        mask = np.zeros([self.numberOfVoxelsX, self.numberOfVoxelsY, self.numberOfVoxelsZ])
        mask = mask.astype(bool)

        # create three 3d Matrices, same shape as the mask
        # 1. martix -> distances from origin in x Direction
        # 2. martix -> distances from origin in y Direction
        # 3. martix -> distances from origin in z Direction

        # 1. Distance from origin in X Direction
        distancesX = np.linspace(0, self.simulationBoxSize[0], self.numberOfVoxelsX)
        distancesX = distancesX.reshape(self.numberOfVoxelsX, 1)
        distancesX = np.repeat(distancesX, self.numberOfVoxelsZ, axis=1)
        distancesX = np.expand_dims(distancesX, axis=1)
        distancesX = np.repeat(distancesX, self.numberOfVoxelsY, axis=1) 

        # 2. Distance from origin in Y Direction
        distancesY = np.linspace(0, self.simulationBoxSize[1], self.numberOfVoxelsY)
        distancesY = distancesY.reshape(self.numberOfVoxelsY, 1)
        distancesY = np.tile(distancesY, (self.numberOfVoxelsX, 1, self.numberOfVoxelsZ))
        
        # 3. Distance from origin in Z Direction
        distancesZ = np.linspace(0, self.numberOfVoxelsZ*self.voxelSizeZ-self.voxelSizeZ, self.numberOfVoxelsZ)
        distancesZ = np.tile(distancesZ, (self.numberOfVoxelsX, self.numberOfVoxelsY, 1))

        # create a individual mask for each atom. These masks are joined to a single mask
        for coor in referenceCoordinates:
            distancesToAtom = np.sqrt((distancesX - coor[0])**2 + (distancesY - coor[1])**2 + (distancesZ - coor[2])**2)

            mask += distancesToAtom <= integrationRadius
        
        # calculate the electron density
        electronDensity = np.sum(mask * self.data)

        voxelMatrix = np.array([[self.voxelSizeX, 0, 0],
                               [0, self.voxelSizeY, 0],
                               [0, 0, self.voxelSizeZ]])
        volume = np.linalg.det(voxelMatrix)

        numberOfElectrons = volume * electronDensity

        return numberOfElectrons 