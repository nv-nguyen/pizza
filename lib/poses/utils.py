import numpy as np
import math

_FLOAT_EPS_4 = np.finfo(float).eps * 4.0


def rodrigues(x, y, z):
    matrix = np.eye(3)
    omega_skew = np.zeros((3, 3))
    omega_skew[0, 1] = -z
    omega_skew[1, 0] = z
    omega_skew[0, 2] = y
    omega_skew[2, 0] = -y
    omega_skew[1, 2] = -x
    omega_skew[2, 1] = x

    omega_skew_sqr = np.matmul(omega_skew, omega_skew)
    theta_sqr = x ** 2 + y ** 2 + z ** 2
    theta = math.sqrt(theta_sqr)
    sin_theta = math.sin(theta)

    if theta == 0:
        return np.eye(3)

    one_minus_cos_theta = 1 - math.cos(theta)
    one_minus_cos_div_theta_sqr = one_minus_cos_theta / theta_sqr

    sin_theta_div_theta_tensor = np.ones((3, 3))
    one_minus_cos_div_theta_sqr_tensor = np.ones((3, 3))

    if theta_sqr > 1e-12 and theta != 0:
        sin_theta_div_theta = sin_theta / theta
        sin_theta_div_theta_tensor.fill(sin_theta_div_theta)
        one_minus_cos_div_theta_sqr_tensor.fill(one_minus_cos_div_theta_sqr)
    else:
        sin_theta_div_theta_tensor.fill(1)
        one_minus_cos_div_theta_sqr_tensor.fill(0)
    matrix = matrix + np.multiply(sin_theta_div_theta_tensor, omega_skew) + \
             np.multiply(one_minus_cos_div_theta_sqr_tensor, omega_skew_sqr)
    return matrix


def rodrigues_inverse(matrix):
    """
    based on this post : http://math.stackexchange.com/questions/83874/efficient-and-accurate-numerical-implementation-of-the-inverse-rodrigues-rotatio
    :param matrix: rotation matrix
    :return: x, y, z
    """
    x, y, z = 0, 0, 0
    r = matrix - matrix.T
    t = np.trace(matrix)
    if t >= 3. - 1e-12:
        w = (0.5 - ((t - 3.) / 12.)) * r
        x, y, z = w[2, 1], w[0, 2], w[1, 0]
    elif t > -1. + 1e-12:
        theta = math.acos((t - 1.) / 2.)
        w = (theta / (2. * math.sin(theta))) * r
        x, y, z = w[2, 1], w[0, 2], w[1, 0]
    else:
        diag = np.diag(matrix)
        a = np.argmax(diag)
        b = (a + 1) % 3
        c = (a + 2) % 3
        s = np.sqrt(diag[a] - diag[b] - diag[c] + 1)
        v = np.zeros(diag.shape)
        # unit quaternion (w, v)
        v[a] = s / 2.
        v[b] = (1. / (2. * s)) * (matrix[b, a] + matrix[a, b])
        v[c] = (1. / (2. * s)) * (matrix[c, a] + matrix[a, c])
        v = math.pi * (v / np.linalg.norm(v))
        x, y, z = v[0], v[1], v[2]
    return x, y, z


def euler2quat(z=0, y=0, x=0):
    ''' Return quaternion corresponding to these Euler angles
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format
    Notes
    -----
    We can derive this formula in Sympy using:
    1. Formula giving quaternion corresponding to rotation of theta radians
       about arbitrary axis:
       http://mathworld.wolfram.com/EulerParameters.html
    2. Generated formulae from 1.) for quaternions corresponding to
       theta radians rotations about ``x, y, z`` axes
    3. Apply quaternion multiplication formula -
       http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
       formulae from 2.) to give formula for combined rotations.
    '''
    z = z / 2.0
    y = y / 2.0
    x = x / 2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
        cx * cy * cz - sx * sy * sz,
        cx * sy * sz + cy * cz * sx,
        cx * cz * sy - sx * cy * sz,
        cx * cy * sz + sx * cz * sy])


def mat2xyz_euler(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.
    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)
    Problems arise when cos(y) is close to zero, because both of::
       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33 * r33 + r23 * r23)
    if cy > cy_thresh:  # cos(y) not close to zero, standard form
        z = math.atan2(-r12, r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13, cy)  # atan2(sin(y), cy)
        x = math.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21, r22)
        y = math.atan2(r13, cy)  # atan2(sin(y), cy)
        x = 0.0
    return x, y, z


def get_intrinsic_matrix(dataset_name):
    if dataset_name == 'shapenet':
        width = 640
        height = 360
        focal = 50
        sensor_width = 36
        focal = focal / 1000.0
        pixel_size = sensor_width / (1000.0 * width)
        intrinsic = np.array([[focal / pixel_size, 0, width / 2],
                              [0, focal / pixel_size, height / 2],
                              [0, 0, 1]])
    elif dataset_name == 'modelNet':  # given by DeepIM
        intrinsic = np.array([[572.4114, 0, 320],
                              [0., 572.4114, 180],
                              [0., 0., 1.]])
    elif dataset_name == 'moped':  # given by moped
        intrinsic = np.array(
            [[618.093017578125, 0., 325.8868103027344],
             [0., 617.4840087890625, 252.30706787109375],
             [0., 0., 1]]
        )
    return intrinsic


intrinsic_modelNet = get_intrinsic_matrix('modelNet')
intrinsic_modelNet_inverse = np.linalg.inv(intrinsic_modelNet)


def projection_perspective(list_translations, dataset_name):
    if dataset_name == "modelNet":
        perspective_point = intrinsic_modelNet.dot(list_translations.T).T
    x_coord = perspective_point[:, 0] / perspective_point[:, 2]
    y_coord = perspective_point[:, 1] / perspective_point[:, 2]
    return np.array([x_coord.astype(np.float), y_coord.astype(np.float)]).T


def geodesic_numpy(R1, R2):
    theta = (np.trace(R2.dot(R1.T)) - 1) / 2
    theta = np.clip(theta, -1, 1)
    return np.degrees(np.arccos(theta))
