import numpy as np
from numpy import sin, cos
from scipy.linalg import null_space

def check_array(my_array, rows=3):
    my_array = np.asarray(my_array)
    if my_array.ndim != 2 or  my_array.shape[0] != rows:
        raise ValueError(f'The  array of points should have shape ({rows}, n)')

def stereo_aS3_R3(spoints, c=[0, 0, 0, 0], tol=1.e-10):
    """
    Stereographic projection of the 4d points spoints, 
    from an admissible sphere of center c, and radius=1 to R^3
    
    spoints - array of shape (4, n)
    c -  sphere center; can be different from origin after  an admissible translation
    """
    if c[3] + 1 <=0:
        raise ValueError(f"The sphere  is not an admisible one")
    check_array(spoints, rows=4)
    c = np.asarray(c)
    dist_c = np.linalg.norm(spoints-c.reshape(4,1), axis =0)
    if not np.all(np.fabs(dist_c-1) < tol):
        raise ValueError(f"Given points must lie on a sphere of center {a} ")

    xs, ys, zs, us = spoints
    f = (1+c[3]) / (1+c[3]-us)
    x = c[0] + (xs-c[0])*f
    y = c[1] + (ys-c[1])*f
    z = c[2] + (zs-c[2])*f
    return np.stack((x, y, z))

def inv_stereo_aS3(points, c =[0, 0, 0, 0]):  
    """
    inverse stereographic projection from R^3 to the unit sphere centered at c in R^4
    
    points - array of shape (3, n) - n 3d-points to be lifted to sphere
    c - sphere center
    returns the lifted 4d points onto the unit sphere, as an array of shape (4, n)
    """
    
    check_array(points, rows=3)
    if c[3] + 1 <=0:
        raise ValueError(f"The sphere to project to is not an admisible one")
    x, y, z = points
    f = (2*(c[3]+1)) / ((x-c[0])**2+ (y-c[1])**2+ (z-c[2])**2+(c[3]+1)**2)
    xs = c[0] + f*(x-c[0])
    ys = c[1] + f*(y-c[1])
    zs = c[2] + f*(z-c[2])
    us = c[3] + 1-f * (c[3]+1)
    return np.stack((xs, ys, zs, us))

def setup_rotation(s=0, t=0, U=np.eye(4), tol=1.e-10):
    """
    define an orthogonal 4 x 4 matrix, of determinant 1, 
    using its decomposition Rot = U *C2*U^t, C2=normal form  4d rotation
    
    s, t - radians
    U - orthogonal matrix
    """
    
    if np.linalg.norm(U@U.T-np.eye(4)) > tol:
        raise ValueError(f"Q isn't an orthogonal matrix;eventually adjust tol={tol}")

    C2 = np.array([[cos(s), sin(s), 0, 0],
                   [-sin(s),  cos(s), 0, 0],
                   [0, 0, cos(t), sin(t)],
                   [0, 0, -sin(t),  cos(t)]])
    return (U @ C2) @ U.T


def rotate_sphere(spoints, Rot, c=[0,0,0, 0], tol=1.e-10):
    """
    Apply a rotation Rot to spoints on sphere S(c, 1)
    
    spoints - array of shape (4, n); each column elements represent the coordinates
    of  a point  on the sphere S(c, 1) from R^4
    Rot  an orthogonal 4 x 4 - matrix of det=1

    - rotates the points of the sphere S(c, 1) by Rot*(spoints-c) to get 
      the coordinates of new points on the same sphere
      with respect to the linear frame, Frame =(c; e1, e2, e3)
    - add c to express the points on the rotated sphere with respect
      to the real world Frame=(O, e1, e2, e3)
    """
    
    check_array(spoints, rows=4)
    center = np.asarray(c).reshape(4,1)  
    dist_c = np.linalg.norm(spoints-center, axis =0)
    if not np.all(np.fabs(dist_c-1) < tol):
        raise ValueError(f"Given points must lie on a sphere of center {c} ")
    return np.dot(Rot, spoints-center) + center


def sphere_translate(spoints, c=[0, 0, 0, 0], v=[0, 0, 0, 0], tol=1.e-10):
    """
    translate spoints on the unit sphere from R^4, by an admissible translation, v,
    the translated points belong to the sphere S(c+v, 1)
    """
    
    check_array(spoints, rows=4)
    c = np.asarray(c)
    v = np.asarray(v)
    if c[3] + v[3] + 1 <= 0:
        raise ValueError("The north pole after translation should have the  4th coord > 0")
    dist_c = np.linalg.norm(spoints-c.reshape(4,1), axis =0)
    if not np.all(np.fabs(dist_c-1) < tol):
        raise ValueError(f"Given points must lie on a sphere of center {c} ")
    return spoints + v.reshape(4,1), c + v  #c+v is the new sphere center

def ort_gen_rotation(w1, w2):
    """ 
    w1, w2, two arrays of shape (4, ) to define a subspace in R^4
    returns the orthogonal matrix in the orthogonal decomposition
    of the 3d rotation about the subspace span(w1, w2)
    """
    
    if np.asarray(w1).shape != (4,) or np.asarray(w2).shape != (4,):
        raise ValueError('Your vectors shoud have the shape (4,)')
    
    A = np.vstack((w1, w2))
    if np.linalg.matrix_rank(A) != 2:
        raise ValueErrror('Your vectors are not linear independent')
    # orthonormalize the basis (w1, w2)    
    q, _ = np.linalg.qr(A.T)  
    #def an orthonormal basis in the orthogonal complem to span(w1, w2)
    u3, u4 = null_space(A).T  
    U = np.stack((q[:, 0], q[:, 1], u3, u4), axis=-1) 
    return U
    
def Mobius3d_trans (points,   Rot = np.eye(4), c=[0, 0, 0, 0], 
                    v = [0,0,0,0], tol =1.e-10):
    """
    Defines Mobius transformation from  R^3 to itsself, as a
    composition of three or four maps: 
    inv stereo -> sphere rot -> (eventually sphere transl) -> stereo-proj
    
    points - array of shape (3, n); n 3D points to be transformed
    c - sphere center on which  the points are to be lifted
    Rot - an orthogonal 4 x 4 matrix, of det=1, to define the sphere  rotation
    v -  4D vector that gives the direction of sphere  translation 
    returns an array of shape (3, n) representing the transformed points
    """
    
    check_array(points, rows=3)

    if np.linalg.norm((Rot @ Rot.T)-np.eye(4)) >  tol or abs(np.linalg.det(Rot)-1) > tol:
        raise ValueError(f'Rot must be an orthogonal matrix of det=1;'+\
                           f'redefine or adjust tol={tol} kwarg')

    spoints = inv_stereo_aS3(points, c=c) 

    new_spoints= rotate_sphere(spoints, Rot, c=c) 
    if not (len(list(set(v))) == 1 and 0 in v):
        new_spoints, c = sphere_translate(new_spoints, c=c, v=v) 
    return stereo_aS3_R3(new_spoints, c=c) 
