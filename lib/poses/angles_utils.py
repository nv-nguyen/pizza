import numpy as np
from mathutils import Euler, Matrix, Vector
from scipy.linalg import logm
import math
from functools import reduce

_FLOAT_EPS_4 = np.finfo(float).eps * 4.0


def pose2euler(R, t, opencv=True):
    """
    #https://github.com/thodan/bop_toolkit/blob/53150b649467976b4f619fbffb9efe525c7e11ca/bop_toolkit_lib/view_sampler.py#L195-L236
    #https://github.com/thodan/bop_toolkit/blob/master/scripts/calc_gt_distribution.py#L58
    """
    cam_orig_m = -np.linalg.inv(R).dot(t)
    azimuth = math.atan2(cam_orig_m[0, 0], cam_orig_m[1, 0])
    a = np.linalg.norm(cam_orig_m)
    b = np.linalg.norm([cam_orig_m[0, 0], cam_orig_m[1, 0], 0])
    elevation = math.acos(b / a)

    # Azimuth from [0, 360].
    if azimuth < 0:
        azimuth += 2.0 * math.pi
    azimuth = azimuth * 180. / np.pi
    # Elevation from [-90, 90].
    elevation = elevation * 180. / np.pi
    if cam_orig_m[2, 0] < 0:
        elevation = -elevation
    inplane = rotation2inplane(R, cam_orig_m.reshape(-1), opencv) * 180. / np.pi
    return azimuth, elevation, inplane


def vertex2matrix(vertex):
    """Compute rotation matrix from viewpoint vertex """
    # https://github.com/thodan/bop_toolkit/blob/53150b649467976b4f619fbffb9efe525c7e11ca/bop_toolkit_lib/view_sampler.py#L212
    f = -np.array(vertex)  # Forward direction.
    f /= np.linalg.norm(f)
    u = np.array([0.0, 0.0, 1.0])  # Up direction.
    s = np.cross(f, u)  # Side direction.
    if np.count_nonzero(s) == 0:
        # f and u are parallel, i.e. we are looking along or against Z axis.
        s = np.array([1.0, 0.0, 0.0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)  # Recompute up.
    R = np.array([[s[0], s[1], s[2]],
                  [u[0], u[1], u[2]],
                  [-f[0], -f[1], -f[2]]])
    return R


def rotation2inplane(R, camera_origin, opencv=True):
    rot = vertex2matrix(camera_origin)
    # # Convert from OpenGL to OpenCV coordinate system.
    if opencv:
        R_yz_flip = np.asarray(Euler((-np.pi, 0.0, 0.0)).to_matrix())[:3, :3]
        R = R_yz_flip.dot(R)
    angle_axis = logm(np.matmul(R, rot.T))
    return angle_axis[1, 0]


def spherical2scartesian(azimuth, elevation, r):
    loc_y = r * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))
    loc_x = r * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))
    loc_z = r * np.sin(np.radians(elevation))
    return np.array([loc_x, loc_y, loc_z]).T


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


def xyz_euler2mat(x=0, y=0, z=0):
    ''' Return matrix for rotations around z, y and x axes
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
    M : array shape (3,3)
       Rotation matrix giving same rotation as for given angles
    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True
    The output rotation matrix is equal to the composition of the
    individual rotations
    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True
    You can specify rotations by named arguments
    >>> np.all(M3 == euler2mat(x=xrot))
    True
    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.
    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)
    Rotations are counter-clockwise.
    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True
    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    '''
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


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


def euler2angle_axis(z=0, y=0, x=0):
    ''' Return angle, axis corresponding to these Euler angles
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
    theta : scalar
       angle of rotation
    vector : array shape (3,)
       axis around which rotation occurs
    Examples
    --------
    >>> theta, vec = euler2angle_axis(0, 1.5, 0)
    >>> print(theta)
    1.5
    >>> np.allclose(vec, [0, 1, 0])
    True
    '''
    # delayed import to avoid cyclic dependencies
    import nibabel.quaternions as nq
    return nq.quat2angle_axis(euler2quat(z, y, x))


def angle_axis2euler(theta, vector, is_normalized=False):
    ''' Convert angle, axis pair to Euler angles
    Parameters
    ----------
    theta : scalar
       angle of rotation
    vector : 3 element sequence
       vector specifying axis for rotation.
    is_normalized : bool, optional
       True if vector is already normalized (has norm of 1).  Default
       False
    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively
    Examples
    --------
    >>> z, y, x = angle_axis2euler(0, [1, 0, 0])
    >>> np.allclose((z, y, x), 0)
    True
    Notes
    -----
    It's possible to reduce the amount of calculation a little, by
    combining parts of the ``angle_axis2mat`` and ``mat2euler``
    functions, but the reduction in computation is small, and the code
    repetition is large.
    '''
    # delayed import to avoid cyclic dependencies
    import nibabel.quaternions as nq
    M = nq.angle_axis2mat(theta, vector, is_normalized)
    return mat2xyz_euler(M)


def combine_view_transform(vp, view_transform):
    """
    combines a camera space transform with a camera axis dependent transform.
    Whats important here is that view transform's translation represent the displacement from
    each axis, and rotation from each axis. The rotation is applied around the translation point of view_transform.
    :param vp:
    :param view_transform:
    :return:
    """
    camera_pose = vp.copy()
    R = camera_pose.rotation
    T = camera_pose.translation
    rand_R = view_transform.rotation
    rand_T = view_transform.translation

    rand_R.combine(R)
    T.combine(rand_R)
    rand_T.combine(T)
    return rand_T


def opencv2opengl(cam_matrix_world):
    from scipy.spatial.transform import Rotation as R
    rot180x = R.from_euler('x', 180, degrees=True).as_matrix()
    rotation = cam_matrix_world[:3, :3]
    translation = cam_matrix_world[:3, 3]
    output = np.copy(cam_matrix_world)
    output[:3, :3] = np.asarray(Matrix(rot180x) @ Matrix(rotation).to_3x3())
    output[:3, 3] = np.asarray(Matrix(rot180x) @ Vector(translation))
    return output


def inverse_matrix_world(matrix_4x4):
    rotation = matrix_4x4[:3, :3]
    translation = matrix_4x4[:3, 3]
    r_transpose_x = rotation[0, 0] * translation[0] + rotation[1, 0] * translation[1] + rotation[2, 0] * translation[2]
    r_transpose_y = rotation[0, 1] * translation[0] + rotation[1, 1] * translation[1] + rotation[2, 1] * translation[2]
    r_transpose_z = rotation[0, 2] * translation[0] + rotation[1, 2] * translation[1] + rotation[2, 2] * translation[2]
    matrix_world_inverse = np.array([
        [rotation[0, 0], rotation[1, 0], rotation[2, 0], -r_transpose_x],
        [rotation[0, 1], rotation[1, 1], rotation[2, 1], -r_transpose_y],
        [rotation[0, 2], rotation[1, 2], rotation[2, 2], -r_transpose_z],
        [0, 0, 0, 1.0]])
    return matrix_world_inverse


def apply_matrix_world_to_camera_having_rotation_90x(cam_matrix_world):
    """
    As object is usually imported with euler_X = 90 in Blender, it should be interesting to use this function to have
    a easy visualization
    """
    from scipy.spatial.transform import Rotation as R
    rot90x = R.from_euler('x', 90, degrees=True).as_matrix()

    rotation = cam_matrix_world[:3, :3]
    translation = cam_matrix_world[:3, 3]
    cam_matrix_world[:3, :3] = np.asarray(Matrix(rot90x) @ Matrix(rotation).to_3x3())
    cam_matrix_world[:3, 3] = np.asarray(Matrix(rot90x) @ Vector(translation))
    return cam_matrix_world


def rotationMatrixToEulerAngles(batch_matrix_rotation, degree=True):
    """
    :param degree:
    :param batch_matrix_rotation: Bx3x3 list matrix of rotation
    :return: euler angles corresponding to each matrix
    """
    batch_size = len(batch_matrix_rotation)
    error_euler = np.zeros((batch_size, 3))
    import math
    for i in range(batch_size):
        R = batch_matrix_rotation[i]
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        error_euler[i] = [x, y, z]
    if degree:
        error_euler = np.rad2deg(error_euler)
    return error_euler


def apply_rot180y(pose):
    pose[0, 3] = -pose[0, 3]
    pose[2, 3] = -pose[2, 3]

    from scipy.spatial.transform import Rotation as R
    rot180y = R.from_euler('y', 180, degrees=True).as_matrix()
    pose[:3, :3] = np.dot(rot180y, pose[:3, :3])
    return pose


def apply_rot90x(pose):
    from scipy.spatial.transform import Rotation as R
    rot90x = R.from_euler('x', 90, degrees=True).as_matrix()
    pose[:3, :3] = np.dot(rot90x, pose[:3, :3])
    pose[:3, 3] = np.dot(rot90x, pose[:3, 3])
    return pose


def combine_pose(R, T, S):
    from mathutils import Quaternion
    pose = np.zeros((4, 4))
    pose[:3, :3] = np.asarray(Quaternion(R).to_matrix()) * S[0]  # as we use same scale for each dim
    pose[:3, 3] = T
    return pose


def get_camera_location_from_obj_pose(obj_pose):
    """
    R_tranpose x (-T)
    """
    trans = obj_pose[:3, 3]
    T_cam = obj_pose[:3, :3].T.dot(-trans)
    T_cam = T_cam / np.linalg.norm(T_cam)
    return T_cam


def look_at(location):
    """
    Get object pose from a viewpoint location
    # Taken from https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/view_sampler.py#L216
    IMPORTANT: output of this function is the object pose defined in OPENGL coordinate convention
    """
    f = -np.array(location)  # Forward direction.
    f /= np.linalg.norm(f)

    u = np.array([0.0, 0.0, 1.0])  # Up direction.
    s = np.cross(f, u)  # Side direction.
    if np.count_nonzero(s) == 0:
        # f and u are parallel, i.e. we are looking along or against Z axis.
        s = np.array([1.0, 0.0, 0.0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)  # Recompute up.
    R = np.array([[s[0], s[1], s[2]],
                  [u[0], u[1], u[2]],
                  [-f[0], -f[1], -f[2]]])
    t = - R.dot(np.array(location).reshape((3, 1)))
    obj_pose = np.zeros((4, 4))
    obj_pose[:3, :3] = R
    obj_pose[:3, 3] = -t.reshape(-1)
    obj_pose[3, 3] = 1
    return obj_pose


def remove_inplane_rotation(opencv_pose, return_symmetry_rot=False):
    """
    TODO: this function can be improved and simplified
    """
    cam_location = get_camera_location_from_obj_pose(opencv_pose)
    obj_opengl_pose = look_at(cam_location)
    opencv_pose_wo_inplane = opencv2opengl(obj_opengl_pose)
    opencv_pose_wo_inplane[:3, 3] = opencv_pose[:3, 3]  # do not change the translation
    if return_symmetry_rot:
        opposite_cam_location = cam_location
        opposite_cam_location[:2] *= -1
        obj_opengl_pose_opposite = look_at(opposite_cam_location)
        opencv_pose_wo_inplane_opposite = opencv2opengl(obj_opengl_pose_opposite)
        opencv_pose_wo_inplane_opposite[:3, 3] = opencv_pose[:3, 3]  # do not change the translation
        return opencv_pose_wo_inplane, opencv_pose_wo_inplane_opposite
    else:
        return opencv_pose_wo_inplane


def perspective(K, obj_pose, pts):
    results = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        R, T = obj_pose[:3, :3], obj_pose[:3, 3]
        rep = np.matmul(K, np.matmul(R, pts[i].reshape(3, 1)) + T.reshape(3, 1))
        x = np.int32(rep[0] / rep[2])  # as matplot flip  x axis
        y = np.int32(rep[1] / rep[2])
        results[i] = [x, y]
    return results


def geodesic_numpy(R1, R2):
    theta = (np.trace(R2.dot(R1.T)) - 1) / 2
    theta = np.clip(theta, -1, 1)
    return np.degrees(np.arccos(theta))