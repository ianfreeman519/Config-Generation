from math import *  # Changelog should be attached

protons = int(input("How many protons are in your desired configuration?\n"))
neutrons = int(input("How many neutrons are in your desired configuration?\n"))
nBodies = protons + neutrons
pLeft = protons     # number of protons that still need to be added to the config
nLeft = neutrons    # number of neutrons that still need to be added to the config
minMeshList = []    # will store a list of ordered mesh points to be used for fixing configurations

a = 110     # (MeV) - Strength of short-range repulsion of nucleons
b = -26     # (MeV) - Strength of intermediate-range repulsion of nucleons
c = 24      # (MeV) - Strength of intermediate-range repulsion of nucleons
ul = 1.25   # (fm^2) - Length scale of the nuclear potential (ul represents uppercase llambda)
alpha = 1.4399764   # (C^2) - Elementary charge squared to approximate coulomb repulsion

dMax = 3.0  # the farthest distance from any nucleon the program should solve for potential
dMin = 1.0    # the closest distance from any nucleon the program should solve for potential
dr = 0.2    # the step size for mesh creation (see function "potentialMesh")

visited = [[-1, 0, 0, 0]]   # list that will eventually contain position and psuedo-charge number for visited particles
nLeft = nLeft - 1           # since we used one n to place at origin, we have one less neutron to worry about


def reducemesh(mesh, points):
    """This function iterates through the mesh and the points list (going to be visited list), and removes everything
    within the dMin range and outside the dMax range.  This will hopefully reduce the size of the mesh list and
    will speed up the program."""

    newmesh = mesh

    for p in points:
        for mp in mesh:
            r = sqrt((p[1] - mp[1]) ** 2 + (p[2] - mp[2]) ** 2 + (p[3] - mp[3]) ** 2)
            if r < dMin or r > dMax:
                newmesh.remove(mp)

    return newmesh


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


def potentialMesh(nucleons, q, n):
    """This function creates a mesh of points in space(loop (a)), each 'dr' away from the last (making a mesh density
    of dr^3). It then calculates the potential energy between every point in the nucleons list (usually the visited
    particle list), and every point in the mesh using the potential function v (loop (b)). It returns the coordinate
    point [q, x, y, z] of the minimum potential energy (loop (c)), where q is the psuedo-charge of the desired
    nucleon."""

    global minMeshList
    xmax = 0            # these minimums and maximums need to be defined to develop the bounds of iteration
    xmin = 0
    ymax = 0
    ymin = 0
    zmax = 0
    zmin = 0
    mesh = []           # this empty mesh list will be filled with the mesh points - see loop (a)
    minpot = 999999999999   # minpot starts at 'infinity' so the min pot can be found from the mesh - see loop (c)
    minpoint = -1       # minpoint will be the mesh index that the min potential is found

    for p in range(0, len(nucleons)):   # to determine the dimensions of the mesh, we must first determine extrema
        if xmin >= nucleons[p][1]:
            xmin = nucleons[p][1]
        elif xmax <= nucleons[p][1]:
            xmax = nucleons[p][1]

        if ymin >= nucleons[p][2]:
            ymin = nucleons[p][2]
        elif ymax <= nucleons[p][2]:
            ymax = nucleons[p][2]

        if zmin >= nucleons[p][3]:
            zmin = nucleons[p][3]
        elif zmax <= nucleons[p][3]:
            zmax = nucleons[p][3]

    xmax += dMax          # increasing the region of iteration to ensure mesh captures all potential minimums
    xmin -= dMax
    ymax += dMax
    ymin -= dMax
    zmax += dMax
    zmin -= dMax

    """The following for loop iterates over the bounds determined above, creating an array of mesh points with the form
    [q, x, y, z, 0].  This mesh will later be filled in with [q, x, y, z, potential].  The loops had to be written
    with drange() step intervals because python does not like to read non integer step sizes."""

    # (a) Creating the mesh:
    for x in drange(xmin, xmax, dr):
        for y in drange(ymin, ymax, dr):
            for z in drange(zmin, zmax, dr):
                mesh.append([q, round(x, 4), round(y, 4), round(z, 4), 0])

    mesh = reducemesh(mesh, nucleons)   # to reduce computation time, we ignore close points (see reducemesh)

    testfile = open("testfile.txt", "w+")       # just to figure out whats going wrong
    for n in nucleons:
        testfile.write(str(n[0]) + "\t" + str(n[1]) + "\t" + str(n[2]) + "\t" + str(n[3]) + "\n")
    for mp in mesh:
        testfile.write(str(mp[0]) + "\t" + str(mp[1]) + "\t" + str(mp[2]) + "\t" + str(mp[3]) + "\n")
    testfile.close()

    """The following loop calculates the potentials between every mesh point and particle in the current arrangement, 
    and adds them to the potentials of each mesh point.  If the mesh point and nucleon point coincide, the potential
    is set to a very large number to resemble 'infinite' potential energy."""

    # (b) Calculating the potentials
    for point in range(len(mesh)):
        for nucleon in range(len(nucleons)):
            if mesh[point][1] == nucleons[nucleon][1] and mesh[point][2] == nucleons[nucleon][2] and \
                    mesh[point][3] == nucleons[nucleon][3]:
                mesh[point][4] = 999999999
            else:
                mesh[point][4] += v(mesh[point], nucleons[nucleon])

    # (c) Finding the minimum potential
    for point in range(len(mesh)):
        if minpot > mesh[point][4]:
            minpot = mesh[point][4]
            minpoint = point

    # (d) Appending ordered minimum potential list to minMeshList
    for point in range(len(mesh)):
        # Need to include some kind of sorting per the 4th index of mesh to store in minmeshlist!!!
    return mesh[minpoint]


def v(b1, b2):
    """This function determines the potential energy in MeV of two points.  This potential function is derived from
    Dr. Matt Caplan's equations of nuclear potentials surrounding the strong nuclear force.  The potential() takes two
    lists, each with the following format: [psuedo-charge, x, y, z].  The psuedo-charge is 1 for protons and -1 for
    neutrons.  It is used to differentiate neutron-neutron, proton-neutron, and proton-proton interactions which
    simplifies the calculations."""

    """The suffix [0] is the q (psuedo-charge) of a body, [1] is the x position, [2] is the y position, and [3] is the
    z position"""

    r2 = (b1[1] - b2[1]) ** 2 + (b1[2] - b2[2]) ** 2 + (b1[3] - b2[3]) ** 2     # r^2 represents radius squared
    r = sqrt(r2)                                                                # r is in fm
    sign = ((1 + b1[0]) / 2) * ((1 + b2[0]) / 2)        # used for determining if coulomb potential is needed (0 or 1)
    pot = a * exp(-r2 / ul) + (b + b1[0] * b2[0] * c) * exp(-r2 / (2 * ul)) + sign * a / r

    if abs(pot) < 0.0001:       # ensures no ridiculous values and makes formatting cleaner
        return 0
    else:
        return pot


def main():
    global visited, pLeft, nLeft
    config = open("config.txt", "w+")
    config.write(str(nBodies) + "\n")

    for n in range(nLeft + pLeft):
        if pLeft > nLeft:
            q = 1
            pLeft -= 1
        else:
            q = -1
            nLeft -= 1

        visited.append(potentialMesh(visited, q, n))

    for p in visited:
        print(p)
        config.write(str(p[0]) + "\t" + str(p[1]) + "\t" + str(p[2]) + "\t" + str(p[3]) + "\t0\t0\t0\n")

    config.close()


main()
print(minMeshList[1])
