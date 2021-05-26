"""SETTINGS
"""

PERIODIC_BOUNDS = True # Periodic boundary conditions. If false, the edges of
# the grid will show artifacts. If true,
# the first and last row/column will interact

ROWS, COLS = 200,200 # lattice can take on any rectangular shape
UPDATES_PER_FRAME = 2**10 # will speed up the animation

# spin orientations initiated either randomly, or all spins point into the
# same direction.
RANDOM_INITIAL_STATE = False

RANDOM_ANGLE = True # Spin rotations can be stochastic, or always of the same
# magnitude (Pi in this case). Spin rotations will be a spin flip if set to false.

""" LOGIC
"""
import numpy as np

# Not needed, but might be useful
def E(spin_config, J, h, T):
    """ Spin config is a 2d array that carries all angles theta_i of all spins
    i in the lattice. J is the coupling strength, h scales the magnetic field.
    """

    neighboring_spin_sum = 0
    for i in range(len(spin_config) - 1): # addresses rows
        for j in range(len(spin_config[i]) - 1): # addresses points on lattice
            # Add cos(theta_i - theta_{i+1}) to sum for next neighbors
            # Neighbor on the right:
            neighboring_spin_sum += np.cos(spin_config[i][j] - spin_config[i][j + 1])
            # Neighbor below:
            neighboring_spin_sum += np.cos(spin_config[i][j] - spin_config[i + 1][j])

        # Still missing right column:
        neighboring_spin_sum += np.cos(spin_config[i][-1] - spin_config[i + 1][-1])

    # Still missing bottom row:
    for j in range(len(spin_config[-1]) - 1):
        neighboring_spin_sum += np.cos(spin_config[-1][j] - spin_config[-1][j + 1])

    single_spin_sum = 0
    for i in range(len(spin_config)): # addresses rows
        for j in range(len(spin_config[i])): # addresses points on lattice
            # Add cos(theta_i) to sum
            single_spin_sum += np.cos(spin_config[i][j])

    return -J * neighboring_spin_sum - h * single_spin_sum


def get_rand_site(spin_config):
    """ Takes the spin lattice and returns a random site.
    """
    row = np.random.randint(0, len(spin_config))
    col = np.random.randint(0, len(spin_config[row]))

    return row, col

def deltaE(spin_config, row, col, J, h, phi):
    """ Calculates the energy difference induced by rotating the spin at pos.
    (row, col) by phi.
    """
    # Get theta that was picked:
    theta = spin_config[row][col]
    # Get neighboring spins

    neighbor_list = [0,0,0,0]

    if PERIODIC_BOUNDS:
        # Spins located at the boundary will interact with the spin at the
        # opposite boundary of the lattice (periodic boundary condition).
        neighbor_list[0] = spin_config[row-1][col]
        neighbor_list[1] = spin_config[(row+1)%ROWS][col]
        neighbor_list[2] = spin_config[row][col-1]
        neighbor_list[3] = spin_config[row][(col+1)%COLS]

    else:
        # Spins located at the boundary don't have partners on all 4 sides.
        # We have to check if we are on a site at the boundary, and if so,
        # set the interactiong with the (non-existing) neighboring site to zero.

        # Checking row
        if row == 0:
            # Top boundary
            phi, neighbor_list[0] = 0, 0
        elif row == ROWS-1:
            # Bottom boundary
            phi, neighbor_list[1] = 0, 0
        else:
            # No boundary
            neighbor_list[0] = spin_config[row-1][col]
            neighbor_list[1] = spin_config[row+1][col]

        # Checking column
        if col == 0:
            # Left boundary
            phi, neighbor_list[2] = 0, 0
        elif col == COLS-1:
            # Right boundary
            phi, neighbor_list[3] = 0, 0
        else:
            # No boundary
            neighbor_list[2] = spin_config[row][col-1]
            neighbor_list[3] = spin_config[row][col+1]

    # We rotate theta by phi and calculate the energy difference the rotation
    # causes
    energy_diff = 0
    for neighbor in neighbor_list:
        u = theta - neighbor
        energy_diff += np.cos(u + phi) - np.cos(u)

    energy_diff *= -J
    energy_diff += -h*(np.cos(theta + phi) - np.cos(theta))

    return energy_diff



def update(spin_config, J, h, T):
    """ Takes the spin lattice and updates it using the Metropolis algorithm.
    """
    # Choose random site:
    row, col = get_rand_site(spin_config)

    if RANDOM_ANGLE:
        # Choose random angle
        phi = np.random.rand()*2*np.pi
    else:
        phi = np.pi

    # Get previous energy:
    delta = deltaE(spin_config, row, col, J, h, phi)

    #Probability to accept the spin flip:
    prob = np.min([1, np.exp(-1/T * delta)])

    # Random number between 0 and 1:
    rand = np.random.rand()

    if rand <= prob:
        # Flip the spin
        spin_config[row][col] += phi

    spin_config = (spin_config)%(2*np.pi)

    return spin_config

def update_x(x, spin_config, J, h, T):
    """ Updates x times per function call.
    """
    for _ in range(x):
        spin_config = update(spin_config, J, h, T)

    return spin_config

def animate(i):
    """ Handels matplotlib animation.
    """
    global spin_config
    x = UPDATES_PER_FRAME
    ax.clear()

    ax.set_xlabel(COLS)
    ax.set_ylabel(ROWS)

    spin_config = update_x(x, spin_config,
                    J_slider.val,
                    h_slider.val,
                    T_slider.val)

    im = ax.imshow(spin_config,
                vmin=0,
                vmax=2*np.pi,
                cmap="hsv") # cyclic cmap

    if i==1:
        # Draw static colorbar in the first frame
        cbar = plt.colorbar(im, ax=ax, label=r"$\theta$")
        cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        cbar.set_ticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])


spin_config = np.array(np.random.rand(ROWS,COLS)) * 7 # get random values

# chosen at the top, will make the spins point towards theta = 0
if not RANDOM_INITIAL_STATE:
    spin_config *= 0


"""ANIMATION
"""

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

fig, ax = plt.subplots()

plt.subplots_adjust(bottom=0.25)

plt.tick_params(
            axis="x",
            which="both",
            bottom=False,
            labelbottom=False
)
plt.tick_params(
            axis="y",
            which="both",
            left=False,
            labelleft=False
)


# Sliders to set the parameters dynamically
hmax = 20
Jmax = 20
Tmax = 30
#
# h_init = 0
# J_init = 1
# T_init = 15

h_init = 0
J_init = 1
T_init = 0.8816*J_init

h_slider_ax  = fig.add_axes([0.15, 0.1, 0.65, 0.03])
h_slider = Slider(h_slider_ax, 'h', -hmax, hmax, valinit=h_init)

T_slider_ax  = fig.add_axes([0.15, 0.15, 0.65, 0.03])
T_slider = Slider(T_slider_ax, 'T', 1e-1, Tmax, valinit=T_init)

J_slider_ax  = fig.add_axes([0.15, 0.05, 0.65, 0.03])
J_slider = Slider(J_slider_ax, 'J', -Jmax, Jmax, valinit=J_init)

reset_sliders_button_ax = fig.add_axes([0.02, 0.35, 0.2, 0.06])
reset_sliders_button = Button(reset_sliders_button_ax, "Reset Sliders")

reset_config_random_button_ax = fig.add_axes([0.02, 0.45, 0.2, 0.06])
reset_config_random_button = Button(reset_config_random_button_ax,
                                "Random Config")

reset_config_0_button_ax = fig.add_axes([0.02, 0.55, 0.2, 0.06])
reset_config_0_button = Button(reset_config_0_button_ax,
                                r"$\theta = 0$")

reset_config_pi_button_ax = fig.add_axes([0.02, 0.65, 0.2, 0.06])
reset_config_pi_button = Button(reset_config_pi_button_ax,
                                r"$\theta = \pi$")

# Button functions
def reset_config_random(mouse_action):
    global spin_config
    spin_config = np.array(np.random.rand(ROWS,COLS)) * 7 # get random values

def reset_config_0(mouse_action):
    global spin_config
    spin_config = np.zeros((ROWS,COLS))

def reset_config_pi(mouse_action):
    global spin_config
    spin_config = np.ones((ROWS,COLS)) * np.pi

def reset_sliders(mouse_action):
    h_slider.set_val(h_init)
    J_slider.set_val(J_init)
    T_slider.set_val(T_init)

reset_config_0_button.on_clicked(reset_config_0)
reset_config_pi_button.on_clicked(reset_config_pi)
reset_config_random_button.on_clicked(reset_config_random)
reset_sliders_button.on_clicked(reset_sliders)

# Animate
anim = FuncAnimation(ax.figure,
                     animate,
                     interval=1, # ms
                     cache_frame_data=True)


plt.show()
