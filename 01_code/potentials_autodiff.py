import jax.numpy as jnp
import jax

def create_tuple(n, m):
    """
    Create a tuple with n zeros followed by m ones.

    Parameters:
    n (int): The number of zeros.
    m (int): The number of ones.

    Returns:
    tuple: A tuple containing n zeros followed by m ones.
    """
    # Create the tuple using tuple concatenation
    return (0,) * n + (1,) * m


def get_derivs(f,x0,dmax):
    """
    Compute the partial derivatives of the function, `f`, of a two-dimensional
    input evaluated at the point `x0` up to a maximum order, `dmax`.

    Parameters
    ----------
    f : function
        function to take derivatives of
    x0 : jnp.array
        point at whcih to evaluate derivatives
    dmax : int
        maximum order of derivatives

    Returns
    -------
    dict
        Dictionary storing partial derivative values.
    """
    derivs = {(0,0):f(x0)}
    Dn_f = jax.jacrev(f)
    Dn_f_val = Dn_f(x0)
    derivs[(1,0)] = Dn_f_val[create_tuple(1,0)]
    derivs[(0,1)] = Dn_f_val[create_tuple(0,1)]
    for d in range(2,dmax+1):
        Dn_f = jax.jacfwd(Dn_f)
        Dn_f_val = Dn_f(x0)
        for j in range(d+1):
            i = d-j
            derivs[(i,j)] = Dn_f_val[create_tuple(i,j)]
    return derivs

