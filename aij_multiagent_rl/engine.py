from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from PIL.Image import Image as ImageType
from scipy.ndimage import uniform_filter
from skimage import draw
from skimage.measure import block_reduce

COLOR_MAPS = {
    'red': np.array([255, 1, 1]),
    'blue': np.array([1, 1, 255]),
    'purple': np.array([128, 1, 128]),
    'pink': np.array([255, 200, 255]),
    'yellow': np.array([255, 255, 100]),
    'orange': np.array([235, 155, 1]),
    'gray': np.array([128, 128, 128]),
    'turquoise': np.array([1, 219, 255]),
    'domestic': np.array([255, 1, 1]),
    'foreign': np.array([51, 51, 51]),
}


def create_borders_candidate(
    grid_size: int,
    border_distort_range: Tuple[int, int],
    max_edge_dev: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Create candidate border split

    Create candidate for agents' segments randomised split

    Args:
        grid_size: 2D square game field size
        border_distort_range: noise range for borders distortion
        max_edge_dev: maximum segment border shift
        rng: numpy random number generator

    Returns:
        Tuple[np.ndarray, np.ndarray, bool]: tuple of
            (horizontal borders, vertical borders, validity flag)
    """
    h_borders = np.zeros((grid_size, grid_size))
    v_borders = np.zeros((grid_size, grid_size))
    border_steps = np.linspace(0, grid_size, 4)[1:-1].astype(int)
    low, high = border_distort_range
    valid = True

    # horizontal borders
    h_inds = []
    for bs in border_steps:
        h_bord = rng.integers(low=low, high=high, size=(grid_size,))
        h_bord = np.cumsum(h_bord)
        h_bord += bs
        h_inds.append(h_bord)
        h_borders[h_bord, range(h_borders.shape[1])] = 1
        # check for drift
        if (np.abs(h_bord[-1] - bs) / grid_size) > max_edge_dev:
            valid = False

    # Border overlap
    if max(h_inds[0]) > min(h_inds[1]):
        valid = False

    # vertical borders
    v_inds = []
    for bs in border_steps:
        v_bord = rng.integers(low=low, high=high, size=(grid_size,))
        v_bord = np.cumsum(v_bord)
        v_bord += bs
        v_inds.append(v_bord)
        v_borders[range(v_borders.shape[0]), v_bord] = 1
        # check for drift
        if (np.abs(v_bord[-1] - bs) / grid_size) > max_edge_dev:
            valid = False

    # Border overlap
    if max(v_inds[0]) > min(v_inds[1]):
        valid = False

    return h_borders, v_borders, valid


def create_borders(
    grid_size: int,
    border_distort_range: Tuple[int, int],
    max_edge_dev: float,
    rng: np.random.Generator,
    max_tries: Optional[int] = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Create valid borders

    Create valid borders with multiple randomized attempts
    until success

    Args:
        grid_size: 2D square game field size
        border_distort_range: noise range for borders distortion
        max_edge_dev: maximum segment border shift
        rng: numpy random number generator
        max_tries: maximum attempts for border generation

    Returns:
        Tuple[np.ndarray, np.ndarray]: tuple of
            (horizontal borders, vertical borders)
    """
    valid = False
    n_tries = 0
    h_borders, v_borders = None, None
    while not valid:
        h_borders, v_borders, valid = create_borders_candidate(
            grid_size=grid_size,
            border_distort_range=border_distort_range,
            max_edge_dev=max_edge_dev,
            rng=rng
        )
        n_tries += 1
        if n_tries > max_tries:
            raise AssertionError(
                f'Exceeded max number of generation attempts: {max_tries}'
            )
    return h_borders, v_borders


def get_segment_map(
    h_borders: np.ndarray,
    v_borders: np.ndarray
) -> Dict[str, np.ndarray]:
    """Create segment map

    Create valid segment split for simulation

    Args:
        h_borders: np.ndarray of horizontal borders
        v_borders: np.ndarray of vertical borders

    Returns:
        Dict[str, np.ndarray]: dictionary of binary masks
            for each segment on the map
    """
    h_csum = np.cumsum(h_borders, axis=0)
    v_csum = (np.cumsum(v_borders, axis=1) + 1) * 10
    raw_segment_map = (h_csum + v_csum).astype(int)
    seg_ids = sorted(np.unique(raw_segment_map))
    agent_id = 0
    segment_map = {}
    for s in seg_ids:
        if s != 21:  # non water case
            segment_map[f'agent_{agent_id}'] = (
                    raw_segment_map == s).astype(int)
            agent_id += 1
        else:  # water case
            segment_map['water'] = (raw_segment_map == s).astype(int)
    return segment_map


def get_machines_candidate(
    s_map: np.ndarray,
    distance: int,
    machine_size: int,
    rng: np.random.Generator
) -> Tuple[np.ndarray, ...]:
    """Create candidate placement for machines

    Create candidate placement for machines for given
    individual agent segment

    Args:
        s_map: binary segment map for a given agent
        distance: initial distance between fabricator and recycler
        machine_size: machine icon size in pixels
        rng:  numpy random number generator

    Returns:
        Tuple[np.ndarray, ...]: tuple of numpy arrays which has
            - fabricator center location
            - recycler center location
            - fabricator binary map
            - recycler binary map
            - segment center location
    """
    center = (np
              .argwhere(s_map == 1)
              .mean(axis=0)
              .round(0)
              .astype(int))
    rad = rng.uniform(0, 2 * np.pi)
    h_comp = distance * np.cos(rad)
    v_comp = distance * np.sin(rad)
    machine_loc = np.array(
        [center[0] + h_comp, center[1] + v_comp]
    ).round(0).astype(int)
    recycler_loc = np.array(
        [center[0] - h_comp, center[1] - v_comp]
    ).round(0).astype(int)
    # machines loc
    s = machine_size // 2
    ms = machine_size
    machine_map = np.pad(
        np.zeros_like(s_map), ms, constant_values=0
    ).astype(int)
    recycler_map = np.pad(
        np.zeros_like(s_map), ms, constant_values=0
    ).astype(int)
    machine_map[(machine_loc[0] - s + ms):(machine_loc[0] + s + ms + 1),
                (machine_loc[1] - s + ms):(machine_loc[1] + s + ms + 1)] = 1
    recycler_map[(recycler_loc[0] - s + ms):(recycler_loc[0] + s + ms + 1),
                 (recycler_loc[1] - s + ms):(recycler_loc[1] + s + ms + 1)] = 1
    return machine_loc, recycler_loc, machine_map, recycler_map, center


def validate_locs(
    s_map: np.ndarray,
    machine_size: int,
    machine_map: np.ndarray,
    recycler_map: np.ndarray
) -> bool:
    """Validate machines placement

    Validate machines placement

    Args:
        s_map: binary segment map for a given agent
        machine_size: machine icon size in pixels
        machine_map: fabricator binary map
        recycler_map: recycler binary map

    Returns:
        bool: machines placement validity
    """
    s_map_pad = np.pad(
        s_map, machine_size, constant_values=0
    ).astype(int)
    negative_map = np.logical_not(s_map_pad)
    valid = True
    total_map = np.logical_or(machine_map, recycler_map)
    if np.logical_and(negative_map, total_map).sum() > 0:
        # check for borders violation
        valid = False
    if np.logical_and(machine_map, recycler_map).sum() > 0:
        # check for no intersection between
        valid = False
    return valid


def get_machines_loc(
    s_map: np.ndarray,
    machine_size: int,
    rng: np.random.Generator
) -> Tuple[np.ndarray, ...]:
    """Create placement for machines

    Create placement for machines for a given individual agent
    segment

    Args:
        s_map: binary segment map for a given agent
        machine_size: machine icon size in pixels
        rng:  numpy random number generator

    Returns:
        Tuple[np.ndarray, ...]: tuple of numpy arrays which has
            - fabricator center location
            - recycler center location
            - fabricator binary map
            - recycler binary map
            - segment center location
    """
    distance = s_map.shape[0] // 6
    valid = False
    while not valid and distance > 0:
        locs = get_machines_candidate(
            s_map=s_map, machine_size=machine_size,
            distance=distance, rng=rng
        )
        machine_loc, recycler_loc, machine_map, recycler_map, center = locs
        valid = validate_locs(
            s_map=s_map, machine_size=machine_size,
            machine_map=machine_map, recycler_map=recycler_map
        )
        distance -= 1
    if not valid:
        raise AssertionError(
            'Valid machines locations not found'
        )
    # get rid of padding for maps
    p = machine_size
    sum_pad = np.logical_or(machine_map, recycler_map).sum()
    machine_map = machine_map[p:-p, p:-p]
    recycler_map = recycler_map[p:-p, p:-p]
    sum_crop = np.logical_or(machine_map, recycler_map).sum()
    assert sum_pad == sum_crop, 'crop is invalid'
    return machine_loc, recycler_loc, machine_map, recycler_map, center


def get_all_machines_loc(
    segment_map: Dict[str, np.ndarray],
    machine_size: int,
    rng: np.random.Generator
) -> Tuple[Dict[str, Any], ...]:
    """Create placement for machines

    Create placement for machines for all agents segments

    Args:
        segment_map: dictionary with binary segment maps for each agent
        machine_size: machine icon size in pixels
        rng:  numpy random number generator

    Returns:
        Tuple[Dict[str, Any], ...]: dictionaries with information
            about fabricators, recyclers and segment centers placement
    """
    machines, recyclers, centers = {}, {}, {}
    for seg, s_map in segment_map.items():
        if seg != 'water':
            ml, rl, mm, rm, c = get_machines_loc(
                s_map=s_map, machine_size=machine_size,
                rng=rng
            )
            machines[seg] = {}
            recyclers[seg] = {}
            centers[seg] = c
            machines[seg]['center'] = ml
            machines[seg]['map'] = mm
            recyclers[seg]['center'] = rl
            recyclers[seg]['map'] = rm
    return machines, recyclers, centers


def get_non_resource_regions(
    segment_map: Dict[str, np.ndarray],
    machine_size: int,
    machines: Dict[str, Dict[str, np.ndarray]],
    recyclers: Dict[str, Dict[str, np.ndarray]],
) -> np.ndarray:
    """Get non resource regions

    Get binary map for regions where resources can
    not be located

    Args:
        segment_map: dictionary with binary segment maps for each agent
        machine_size: machine icon size in pixels
        machines: placement information for fabricators
        recyclers: placement information for recyclers

    Returns:
        np.ndarray: binary map for regions where resources can
            not be located
    """
    non_resource = segment_map['water'].copy()
    for a in machines.keys():
        mm = machines[a]['map']
        rm = recyclers[a]['map']
        am = np.logical_or(mm, rm)
        non_resource = np.logical_or(non_resource, am)
    non_resource = uniform_filter(
        non_resource.astype(float),
        size=int(machine_size),
        mode='constant'
    )
    return (non_resource > 1e-5).astype(np.uint8)


def get_region_grid(
    region: np.ndarray,
    cell_size: int,
    agg_fn: Optional = np.max
) -> np.ndarray:
    """Aggregate 2d array

    Aggregate 2d array with a given agg_func

    Args:
        region: region numerical representation
        cell_size: 2D aggregation window size
        agg_fn: function to aggregate with

    Returns:
        np.ndarray: aggregated region representation
    """
    region_grid = block_reduce(
        region,
        (cell_size, cell_size),
        agg_fn
    )
    return region_grid


def get_segment_map_grid(
    segment_map: Dict[str, np.ndarray],
    cell_size: int,
) -> Dict[str, np.ndarray]:
    """Aggregate segment map

    Aggregate binary segment map

    Args:
        segment_map: dictionary with binary segment maps for each agent
        cell_size: 2D aggregation window size

    Returns:
        Dict[str, np.ndarray]: dictionary with aggregated
            binary segment maps for each agent
    """
    grid_segment_map = {}
    for k, v in segment_map.items():
        grid_segment_map[k] = (get_region_grid(
            region=v, cell_size=cell_size,
            agg_fn=np.mean
        ) > 0.5).astype(np.uint8)
    return grid_segment_map


def get_unreachable_regions(
    segment_map: Dict[str, np.ndarray],
    machines: Dict[str, Dict[str, np.ndarray]],
    recyclers: Dict[str, Dict[str, np.ndarray]],
    include_water: Optional[bool] = True
) -> np.ndarray:
    """Get unreachable regions

    Get regions that cannot be accessed  by agents

    Args:
        segment_map: dictionary with binary segment maps for each agent
        machines: placement information for fabricators
        recyclers: placement information for recyclers
        include_water: whether to include water segment

    Returns:
        np.ndarray: binary map with unreachable regions
    """
    all_machines = np.zeros_like(segment_map['water'])
    for a in machines.keys():
        mm = machines[a]['map']
        rm = recyclers[a]['map']
        am = np.logical_or(mm, rm)
        all_machines = np.logical_or(all_machines, am)
    if include_water:
        unreachable = np.logical_or(all_machines, segment_map['water'])
    else:
        unreachable = all_machines
    return unreachable.astype(np.uint8)


def spawn_resources_grid(
    non_resource_grid: np.ndarray,
    resource_prob: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Spawn initial resources

    Spawn initial resources with uniform probability

    Args:
        non_resource_grid: binary map for regions where resources can
            not be located
        resource_prob: probability to spawn resource at a given point
        rng: numpy random numbers generator

    Returns:
        np.ndarray: binary grid with spawned resources
    """
    init_resource_prob_map = rng.uniform(size=non_resource_grid.shape)
    init_resource_prob_map += non_resource_grid
    resources_grid = (init_resource_prob_map < resource_prob)
    return resources_grid.astype(np.uint8)


def grid_to_center_mappings(
    grid: np.ndarray,
    cell_size: int
) -> Tuple[Dict[tuple, np.ndarray], ...]:
    """Map grid to actual playing field

    Map low dimensional grid to actual playing field

    Args:
        grid: low dimensional grid
        cell_size: grid cell size on actual map

    Returns:
        Tuple[Dict[tuple, np.ndarray], ...]: mappings from
            grid coordinates to actual map coordinates and
            vise verse
    """
    grid_loc = np.argwhere(grid > 0)
    center_loc = grid_loc * cell_size + cell_size // 2
    grid_to_center = {tuple(g): c for g, c in zip(grid_loc, center_loc)}
    center_to_grid = {tuple(c): g for g, c in zip(grid_loc, center_loc)}
    return grid_to_center, center_to_grid


def get_template_texture(
    cmap: np.ndarray,
    template: np.ndarray,
    color_eps: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Get template texture

    Get template texture for map visualisation

    Args:
        cmap: background colormap (grass)
        template: map template for dimensions
        color_eps: color noise
        rng: numpy random numbers generator

    Returns:
        np.ndarray: array with background texture
    """
    color_noise = rng.uniform(
        low=-color_eps, high=color_eps,
        size=template.shape
    ) * 255.
    color_noise = np.round(color_noise, 0)[:, :, np.newaxis]
    cmap = cmap[np.newaxis, np.newaxis, :]
    texture = cmap + color_noise
    texture = np.clip(texture, a_min=0, a_max=255)
    texture *= template[:, :, np.newaxis]
    return texture.astype(int)


def get_thick_border(
    borders: np.ndarray,
    border_display_width: int
) -> np.ndarray:
    """Get thick segment borders

    Get thick segment borders

    Args:
        borders: array with segments borders binary map
        border_display_width: segments borders thickness

    Returns:
        np.ndarray: array with segment borders
    """
    thick_borders = uniform_filter(
        borders.astype(float),
        size=border_display_width,
        mode='constant'
    )
    return (thick_borders > 1e-5).astype(int)


def get_machine_icon(
    cmap: np.ndarray,
    asset_size: int
) -> ImageType:
    """Get fabricator icon

    Get icon for fabricator

    Args:
        cmap: asset's color map
        asset_size: asset's size in pixels

    Returns:
        ImageType: asset PIL image
    """
    gen_size = 51
    outline = 5
    cmap = cmap[np.newaxis, np.newaxis, :]
    machine = np.zeros(shape=(gen_size, gen_size, 3))
    machine += cmap

    vec_a = np.abs(np.arange(0, gen_size) - gen_size // 2)
    vec_a -= vec_a.max()
    vec_b = vec_a
    dot = np.matmul(vec_a[:, np.newaxis], vec_b[np.newaxis, :])
    dot = np.abs(dot / dot.max())
    dot = (dot < 0.5).astype(int)
    dot = dot[:, :, np.newaxis]

    machine = machine * dot
    machine[:outline, :, :] = 0
    machine[-outline:, :, :] = 0
    machine[:, :outline, :] = 0
    machine[:, -outline:, :] = 0
    machine = machine.astype(np.uint8)
    image = Image.fromarray(machine)
    machine_icon = image.resize(size=(asset_size, asset_size))
    return machine_icon


def create_circular_mask(
    h: int, w: int,
    center: Optional = None, radius: Optional = None
):
    """Create circular mask
    Args:
        h: height
        w: width
        center: center
        radius: radius
    Returns:
        np.ndarray: circular mask numpy array
    """
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def get_recycler_icon(
    cmap: np.ndarray,
    asset_size: int
) -> ImageType:
    """Get recycler icon

    Get icon for recycler

    Args:
        cmap: asset's color map
        asset_size: asset's size in pixels

    Returns:
        ImageType: asset PIL image
    """
    gen_size = 51
    mask = create_circular_mask(
        gen_size, gen_size)[:, :, np.newaxis]
    cmap = cmap[np.newaxis, np.newaxis, :]
    mask = cmap * mask
    mask = mask.astype(np.uint8)
    image = Image.fromarray(mask)
    recycler_icon = image.resize(size=(asset_size, asset_size))
    return recycler_icon


def get_resource_icon(
    asset_size: int,
    cmap: np.ndarray = np.array([220, 255, 255]),
    tolerance: int = 20
) -> ImageType:
    """Get resource icon

    Get icon for resource

    Args:
        cmap: asset's color map
        asset_size: asset's size in pixels
        tolerance: minimum pixel value to display

    Returns:
        ImageType: asset PIL image
    """
    img = Image.new('RGB', (51, 51))
    img = np.array(img)
    row1, col1 = draw.polygon((1, 50, 35, 50), (25, 10, 25, 40))
    row2, col2 = draw.polygon((20, 20, 34), (0, 50, 25))
    img[row1, col1, :] = cmap
    img[row2, col2, :] = cmap
    img = Image.fromarray(img)
    img = np.array(
        img.resize(size=(asset_size, asset_size))
    )
    filter = np.argwhere(img.mean(axis=2) < tolerance)
    img[filter[:, 0], filter[:, 1], :] = 0
    return Image.fromarray(img)


def get_trash_icon(
    asset_size: int,
    cmap: np.ndarray = np.array([94, 45, 1])
) -> ImageType:
    """Get trash icon

    Get icon for trash

    Args:
        cmap: asset's color map
        asset_size: asset's size in pixels

    Returns:
        ImageType: asset PIL image
    """
    img = np.ones(shape=(asset_size, asset_size, 3))
    img = img * cmap[np.newaxis, np.newaxis, :]
    return Image.fromarray(img.astype(np.uint8))


def get_agent_icon(
    asset_size: int,
    cmap: np.ndarray,
    tolerance: int = 20
) -> ImageType:
    """Get agent icon

    Get icon for agent

    Args:
        cmap: asset's color map
        asset_size: asset's size in pixels
        tolerance: minimum pixel value to display

    Returns:
        ImageType: asset PIL image
    """
    img = Image.new('RGB', (51, 51))
    img = np.array(img)
    row1, col1 = draw.polygon((1, 50, 50, 40, 40, 50, 50),
                              (25, 1, 18, 18, 32, 32, 50))
    img[row1, col1, :] = cmap
    img = Image.fromarray(img)
    img = np.array(
        img.resize(size=(asset_size, asset_size))
    )
    filter = np.argwhere(img.mean(axis=2) < tolerance)
    img[filter[:, 0], filter[:, 1], :] = 0
    return Image.fromarray(img)


def get_arrow_icon(
    asset_size: int,
    cmap: np.ndarray,
    tolerance: int = 10
) -> ImageType:
    """Get arrow icon

    Get icon for arrow

    Args:
        cmap: asset's color map
        asset_size: asset's size in pixels
        tolerance: minimum pixel value to display

    Returns:
        ImageType: asset PIL image
    """
    img = Image.new('RGB', (51, 51))
    img = np.array(img)
    row1, col1 = draw.polygon((0, 50, 35, 50), (25, 14, 25, 36))
    img[row1, col1, :] = cmap
    img = Image.fromarray(img)
    img = np.array(
        img.resize(size=(asset_size, asset_size))
    )
    filter = np.argwhere(img.mean(axis=2) < tolerance)
    img[filter[:, 0], filter[:, 1], :] = 0
    return Image.fromarray(img)


def create_icons(
    segment_map: Dict[str, np.ndarray],
    machine_icon_size: int,
    agent_icon_size: int,
    arrow_icon_size: int,
    resource_icon_size: int,
    trash_icon_size: int,
    rng: np.random.Generator,
    color_maps: Dict[str, np.ndarray] = COLOR_MAPS,
    resource_cmap: np.ndarray = np.array([220, 255, 255]),
    trash_cmap: np.ndarray = np.array([94, 45, 1]),
    north_arrow_cmap: np.ndarray = np.array([255, 1, 1]),
    home_arrow_cmap: np.ndarray = np.array([1, 255, 1]),
    non_home_arrow_cmap: np.ndarray = np.array([1, 1, 255]),
) -> Dict[str, Any]:
    """Get icons

    Get all icons required for visualisation

    Args:
        segment_map: dictionary with binary segment maps for each agent
        machine_icon_size: machine icon size in pixels
        agent_icon_size: agent icon size in pixels
        arrow_icon_size: arrow icon size in pixels
        resource_icon_size: resource icon size in pixels
        trash_icon_size: trash icon size in pixels
        rng: numpy random number generator
        color_maps: color maps for each agent
        resource_cmap: resource color map
        trash_cmap: trash color map
        north_arrow_cmap: north arrow color map
        home_arrow_cmap: home arrow color map
        non_home_arrow_cmap: non home arrow color map

    Returns:
        Dict[str, Any]: dictionary with icons for each asset
    """
    icons = {}
    agents = [a for a in segment_map.keys() if a != 'water']
    # rng.shuffle(agents)
    color_reference = {a: c for a, c in zip(agents, color_maps)}
    color_reference['domestic'] = 'domestic'
    color_reference['foreign'] = 'foreign'
    for a, c in color_reference.items():
        icons[a] = {}
        icons[a]['machine'] = get_machine_icon(
            cmap=color_maps[c],
            asset_size=machine_icon_size
        )
        icons[a]['recycler'] = get_recycler_icon(
            cmap=color_maps[c],
            asset_size=machine_icon_size
        )
        icons[a]['agent'] = get_agent_icon(
            cmap=color_maps[c],
            asset_size=agent_icon_size
        )
        icons[a]['cmap'] = color_maps[c]
    icons['resource'] = get_resource_icon(
        cmap=resource_cmap,
        asset_size=resource_icon_size
    )
    icons['trash'] = get_trash_icon(
        cmap=trash_cmap,
        asset_size=trash_icon_size
    )
    icons['north_arrow'] = get_arrow_icon(
        cmap=north_arrow_cmap,
        asset_size=arrow_icon_size
    )
    icons['home_arrow'] = get_arrow_icon(
        cmap=home_arrow_cmap,
        asset_size=arrow_icon_size
    )
    icons['non_home_arrow'] = get_arrow_icon(
        cmap=non_home_arrow_cmap,
        asset_size=arrow_icon_size
    )
    return icons


def create_erasers(
    agent_size: int,
    arrow_size: int,
    resource_size: int,
    trash_size: int
) -> Dict[str, ImageType]:
    """Create erasers

    Create zero-valued templates for removing icons
    from respective image layer

    Args:
        agent_size: agent icon size in pixels
        arrow_size: arrow icon size in pixels
        resource_size: resource icon size in pixels
        trash_size: trash icon size in pixels

    Returns:
        Dict[str, ImageType]: dictionary with
            zero-valued templates for each asset
    """
    erasers = {}
    erasers['agent'] = Image.fromarray(
        np.zeros(shape=(agent_size, agent_size))
    )
    erasers['arrow'] = Image.fromarray(
        np.zeros(shape=(arrow_size, arrow_size))
    )
    erasers['resource'] = Image.fromarray(
        np.zeros(shape=(resource_size, resource_size))
    )
    erasers['trash'] = Image.fromarray(
        np.zeros(shape=(trash_size, trash_size))
    )
    return erasers


def create_substrate_texture(
    segment_map: Dict[str, np.ndarray],
    borders: np.ndarray,
    border_display_width: int,
    machines: Dict[str, np.ndarray],
    recyclers: Dict[str, np.ndarray],
    icons: Dict[str, Any],
    rng: np.random.Generator
) -> np.ndarray:
    """Create substrate texture

    Create substrate texture for state visualisation

    Args:
        segment_map: dictionary with binary segment maps for each agent
        borders: array with segments borders binary map
        border_display_width: segments borders thickness
        machines: placement information for fabricators
        recyclers: placement information for recyclers
        icons: dictionary with icons for each asset
        rng: numpy random number generator

    Returns:
        np.ndarray: substrate texture image array
    """
    grass_texture = get_template_texture(
        cmap=np.array([1, 128, 1]),
        template=np.logical_not(segment_map['water']),
        color_eps=0.075,
        rng=rng
    )
    water_texture = get_template_texture(
        cmap=np.array([120, 140, 250]),
        template=segment_map['water'],
        color_eps=0.04,
        rng=rng
    )
    thick_borders = get_thick_border(
        borders=borders,
        border_display_width=border_display_width
    )
    borders_texture = get_template_texture(
        cmap=np.array([150, 150, 150]),
        template=thick_borders,
        color_eps=0.075,
        rng=rng
    )

    final_texture = grass_texture + water_texture
    final_texture *= np.logical_not(thick_borders)[:, :, np.newaxis]
    final_texture += borders_texture
    final_texture = Image.fromarray(final_texture.astype(np.uint8))
    # Add machines and recycler assets
    for a in segment_map.keys():
        if a != 'water':
            m_icon = icons[a]['machine']
            m_offset = machines[a]['center'] - (m_icon.size[0] // 2)
            r_icon = icons[a]['recycler']
            r_offset = recyclers[a]['center'] - (r_icon.size[0] // 2)
            final_texture.paste(m_icon, tuple(m_offset)[::-1])
            final_texture.paste(r_icon, tuple(r_offset)[::-1])
    return np.array(final_texture).astype(np.uint8)


def get_agents_perspective(
    grid_size: int,
    segment_map: Dict[str, np.ndarray],
    icons: Dict[str, Any],
    machines: Dict[str, Any],
    recyclers: Dict[str, Any],
    obs_dim: int
) -> Dict[str, np.ndarray]:
    """Get agents perspective

    Get machines attributions for the local view of
    each agent

    Args:
        grid_size: 2D square game field size
        segment_map: dictionary with binary segment maps for each agent
        machines: placement information for fabricators
        recyclers: placement information for recyclers
        icons: dictionary with icons for each asset
        obs_dim: local visual observation size

    Returns:
        Dict[str, np.ndarray]: binary maps for foreign and
            domestic machines for each agent
    """
    p = obs_dim
    machines_by_agent = {}
    valid_agents = [a for a in segment_map.keys() if a != 'water']
    for a1 in valid_agents:
        mba = Image.new('RGB', (grid_size, grid_size))
        for a2 in valid_agents:
            if a1 != a2:
                m_icon = icons['foreign']['machine']
                r_icon = icons['foreign']['recycler']
            else:
                m_icon = icons['domestic']['machine']
                r_icon = icons['domestic']['recycler']
            m_offset = machines[a2]['center'] - (m_icon.size[0] // 2)
            r_offset = recyclers[a2]['center'] - (r_icon.size[0] // 2)
            mba.paste(m_icon, tuple(m_offset)[::-1])
            mba.paste(r_icon, tuple(r_offset)[::-1])
        machines_by_agent[a1] = np.pad(
            np.array(mba), pad_width=((p, p), (p, p), (0, 0)),
            constant_values=0)
    return machines_by_agent


def render_init_resources(
    resource_grid_to_center: Dict[tuple, np.ndarray],
    icons: Dict[str, Any],
    grid_size: int
) -> ImageType:
    """Render initial resources

    Render initial resources distribution

    Args:
        grid_size: 2D square game field size
        icons: dictionary with icons for each asset
        resource_grid_to_center: mapping from low dimension
            resource grid to full size game field

    Returns:
        ImageType: PIL image with initial resources
    """
    resource_map = np.zeros(shape=(grid_size, grid_size, 3))
    resource_map = Image.fromarray(resource_map.astype(np.uint8))
    resources_loc = np.array(list(resource_grid_to_center.values()))
    r_icon = np.array(icons['resource'])
    mask = (r_icon.sum(axis=-1) > 0) * 255.
    mask = (mask).astype(np.uint8)
    mask = Image.fromarray(mask)
    r_icon = Image.fromarray(r_icon.astype(np.uint8))
    for rl in resources_loc:
        offset = rl - (r_icon.size[0] // 2)
        offset = tuple(offset)[::-1]
        resource_map.paste(r_icon, offset, mask=mask)
    return resource_map


def render_wealth(
    money: int,
    block_size: int
) -> ImageType:
    """Render wealth for local menu

    Render wealth for local menu (deprecated)

    Args:
        money: current agent's wealth
        block_size: menu block size

    Returns:
        ImageType: PIL image wealth size
    """
    wealth = Image.new('RGB',
                       (block_size * 2, block_size))
    dw = ImageDraw.Draw(wealth)
    dw.text((1, 1), str(money), fill=(255, 255, 255))
    return wealth


def init_agents(
    centers: Dict[str, np.ndarray],
    block_size: int,
    rng: np.random.Generator
) -> Dict[str, Any]:
    """Init agents state

    Create dictionary with agents initial state

    Args:
        block_size: menu block size
        centers: segment centers locations
        rng: numpy random number generator

    Returns:
        Dict[str, Any]: agents initial state
    """
    agents_state = {}
    dirs = [0, 0.5, 1, 1.5]
    for a, c in centers.items():
        agents_state[a] = {}
        agents_state[a]['loc'] = c
        agents_state[a]['dir_pi'] = rng.choice(dirs, 1).item()
        agents_state[a]['inventory'] = {}
        agents_state[a]['inventory']['resource'] = False
        agents_state[a]['inventory']['trash'] = False
        agents_state[a]['inventory']['trash_source'] = None
        agents_state[a]['wealth'] = 0
        agents_state[a]['last_render_wealth'] = 0
        agents_state[a]['last_wealth_image'] = render_wealth(
            money=0, block_size=block_size
        )
    return agents_state


def pi_to_rad(n_pi: float) -> int:
    """Convert pi to radians

    Convert pi to radian angle for rotations

    Args:
        n_pi: angle in arc notation

    Returns:
        int: angle in radians
    """
    return int((n_pi - 0.5) * 180)


def render_agents(
    agents_state: Dict[str, Any],
    icons: Dict[str, Any],
    grid_size: int,
    local_mode: Optional[bool] = False
) -> np.ndarray:
    """Render agents

    Create image layer with agents

    Args:
        agents_state: agents state dictionary
        grid_size: 2D square game field size
        icons: dictionary with icons for each asset
        local_mode: whether to generate subjective view

    Returns:
        np.ndarray: array with agents image layer
    """
    agents_map = Image.new('RGB', (grid_size, grid_size))
    for a, s in agents_state.items():
        if not local_mode:
            icon = icons[a]['agent']
        else:
            icon = icons['foreign']['agent']
        icon = icon.rotate(pi_to_rad(s['dir_pi']))
        y, x = tuple(s['loc'])
        x, y = x - icon.size[0] // 2, y - icon.size[1] // 2
        agents_map.paste(icon, (x, y))
    return np.array(agents_map)


def loc_to_coord(
    loc: np.ndarray,
    grid_size: int
) -> np.ndarray:
    """Numpy loc to coordinates

    Numpy loc to Cartesian coordinates

    Args:
        loc: array with numpy loc
        grid_size: game field grid size

    Returns:
        np.ndarray: array with Cartesian coordinates
    """
    y, x = tuple(loc)
    y = grid_size - y
    return np.array([x, y])


def cart2pol(x: int, y: int) -> Tuple[float, float]:
    """Cartesian to polar

    Convert Cartesian coordinates to polar coordinates

    Args:
        x: x-coordinate
        y: y-coordinate

    Returns:
        Tuple[float, float]: (rho, phi) - polar coordinates
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def dir_to_shift(dir: float) -> np.ndarray:
    """Direction to shift

    Utility function for building local observation

    Args:
        dir: direction in arc notation

    Returns:
        np.ndarray: array with coordinates shift
    """
    if dir == 0.0:
        shift = np.array([0, 1])
    elif dir == 0.5:
        shift = np.array([-1, 0])
    elif dir == 1.0:
        shift = np.array([0, -1])
    elif dir == 1.5:
        shift = np.array([1, 0])
    else:
        raise ValueError(
            f'Invalid direction for movement: {dir} * pi'
        )
    return shift


class GameEngine:

    def __init__(
        self,
        seed: int,
        grid_size: int = 210,  # should be div by resource_size and trash_size
        obs_dim: int = 60,  # should be divisible by 5
        move_step: int = 7,
        resource_price: int = 10,
        recycle_cost: int = 4,
        border_distort_range: Tuple[int, int] = (-1, 2),
        max_edge_dev: float = 0.1,
        max_tries: int = 25,
        machine_size: int = 9,
        machine_reach: int = 9,
        agent_size: int = 9,
        agent_reach: int = 9,
        resource_size: int = 7,
        trash_size: int = 5,
        resource_prob: float = 0.1,
        border_display_width: int = 2,
        blocked_vanish_alpha: float = 0.25
    ):
        """Game Engine

        Game engine for AIJ Multi-Agent AI competition

        Valid action space (discrete):
            # 0: move forward by `move_step` pixels if possible
            # 1: move left by `move_step` pixels if possible
            # 2: move right by `move_step` pixels if possible
            # 3: move backward by `move_step` pixels if possible
            # 4: pickup resource (if closer than `agent_reach` pixels)
            # 5: pickup trash (if closer than `agent_reach` pixels)
            # 6: throw resource (put into machine if closer than `machine_reach`)
            # 7: throw trash (put into recycler if closer than `machine_reach`)
            # 8: noop

        Parameters:
            seed: random seed for engine
            grid_size: 2D square game field size
            obs_dim: local visual observation size
            move_step: movement step size in pixels
            resource_price: reward given for resource processing
            recycle_cost: cost of recycling trash
            border_distort_range: noise range for borders distortion
            max_edge_dev: maximum segment border shift
            max_tries: maximum attempts for border generation
            machine_size: machine icon size in pixels
            machine_reach: size of machine interaction region
            agent_size: agent icon size in pixels
            agent_reach: agent reach when picking up items
            resource_size: resource icon size in pixels
            trash_size: trash icon size in pixels
            resource_prob: probability to spawn resource at a given point
            border_display_width: segments borders thickness
            blocked_vanish_alpha: blocked segment fogging degree
        """
        assert grid_size % resource_size == 0 and grid_size % trash_size == 0
        assert obs_dim % 5 == 0
        assert obs_dim // 5 >= resource_size + trash_size

        self.rng = np.random.default_rng(seed)
        self.grid_size = grid_size
        self.diag = np.sqrt(2 * grid_size ** 2).item()
        self.move_step = move_step
        self.obs_dim = obs_dim
        self.block_size = obs_dim // 5
        self.resource_price = resource_price
        self.recycle_cost = recycle_cost
        self.border_distort_range = border_distort_range
        self.max_edge_dev = max_edge_dev
        self.max_tries = max_tries
        self.machine_size = machine_size
        self.machine_reach = machine_reach
        self.agent_size = agent_size
        self.agent_reach = agent_reach
        self.arrow_size = obs_dim // 5
        self.resource_size = resource_size
        self.trash_size = trash_size
        self.resource_prob = resource_prob
        self.border_display_width = border_display_width
        self.blocked_vanish_alpha = blocked_vanish_alpha

        # Construct main layout
        self.h_borders, self.v_borders = create_borders(
            grid_size=grid_size,
            border_distort_range=border_distort_range,
            max_edge_dev=max_edge_dev,
            max_tries=max_tries,
            rng=self.rng
        )
        self.borders = np.logical_or(self.h_borders, self.v_borders)
        self.segment_map = get_segment_map(
            h_borders=self.h_borders, v_borders=self.v_borders
        )
        self.machines, self.recyclers, self.centers = get_all_machines_loc(
            segment_map=self.segment_map,
            machine_size=machine_size,
            rng=self.rng
        )
        # Segment maps for resources and trash
        self.resource_segment_map = get_segment_map_grid(
            segment_map=self.segment_map,
            cell_size=resource_size
        )
        self.trash_segment_map = get_segment_map_grid(
            segment_map=self.segment_map,
            cell_size=trash_size
        )

        # Distribute initial resources
        non_resource = get_non_resource_regions(
            segment_map=self.segment_map,
            machine_size=machine_size,
            machines=self.machines,
            recyclers=self.recyclers,
        )
        self.non_resource_grid = get_region_grid(
            region=non_resource,
            cell_size=resource_size
        )
        self.resource_grid = spawn_resources_grid(
            non_resource_grid=self.non_resource_grid,
            resource_prob=resource_prob,
            rng=self.rng
        )
        mappings = grid_to_center_mappings(
            grid=self.resource_grid,
            cell_size=resource_size
        )
        self.resource_grid_to_center, self.resource_center_to_grid = mappings

        # Create grid for trash
        self.non_trash_grid = get_region_grid(
            region=non_resource,
            cell_size=trash_size
        )
        self.trash_grid = np.zeros_like(
            self.non_trash_grid
        ).astype(np.uint8)
        self.trash_grid_to_center = {}
        self.trash_center_to_grid = {}
        self.trash_center_to_source = {}

        # Get unreachable regions
        self.non_reachable = get_unreachable_regions(
            segment_map=self.segment_map,
            machines=self.machines,
            recyclers=self.recyclers,
        )
        self.all_machines = get_unreachable_regions(
            segment_map=self.segment_map,
            machines=self.machines,
            recyclers=self.recyclers,
            include_water=False
        )[:, :, np.newaxis]
        self.all_machines = self.pad_state(self.all_machines)

        # Generate assets for rendering
        self.icons = create_icons(
            segment_map=self.segment_map,
            machine_icon_size=self.machine_size,
            agent_icon_size=self.agent_size,
            arrow_icon_size=self.arrow_size,
            resource_icon_size=self.resource_size,
            trash_icon_size=self.trash_size,
            rng=self.rng
        )

        # Generate erasers for assets
        self.erasers = create_erasers(
            agent_size=self.agent_size,
            arrow_size=self.arrow_size,
            resource_size=self.resource_size,
            trash_size=self.trash_size
        )

        # Generate main substrate texture
        self.substrate_texture = create_substrate_texture(
            segment_map=self.segment_map,
            borders=self.borders,
            border_display_width=border_display_width,
            machines=self.machines,
            recyclers=self.recyclers,
            icons=self.icons,
            rng=self.rng
        )
        self.agents_perspectives = get_agents_perspective(
            grid_size=self.grid_size,
            segment_map=self.segment_map,
            icons=self.icons,
            machines=self.machines,
            recyclers=self.recyclers,
            obs_dim=self.obs_dim
        )

        # Generate resources layer
        self.resource_map = render_init_resources(
            resource_grid_to_center=self.resource_grid_to_center,
            icons=self.icons,
            grid_size=grid_size
        )

        # Generate trash layer
        self.trash_map = Image.new('RGB', (grid_size, grid_size))

        # Create agents state
        self.agents_state = init_agents(
            centers=self.centers,
            block_size=self.block_size,
            rng=self.rng
        )

        # Create blocked dict
        self.blocked = {a: False for a in self.agents_state.keys()}
        self.blocked_map = np.zeros(
            shape=(grid_size, grid_size)).astype(np.uint8)

    def agents_map(self, local_mode: Optional[bool] = False) -> np.ndarray:
        """Render agents

        Create image layer with agents

        Args:
            local_mode: whether to generate subjective view

        Returns:
            np.ndarray: array with agents image layer
        """
        return render_agents(
            agents_state=self.agents_state,
            icons=self.icons,
            grid_size=self.grid_size,
            local_mode=local_mode
        )

    def get_state(self) -> np.ndarray:
        """Get state

        Get current game engine state

        Returns:
            np.ndarray: image array with current visual state
        """
        resource_map = np.array(self.resource_map)
        trash_map = np.array(self.trash_map)
        # add resources layer
        state = self.substrate_texture * np.logical_not(
            resource_map > 0)
        state = state + resource_map
        # add trash layer
        state = state * np.logical_not(
            trash_map > 0)
        state = state + trash_map
        # add agents layer
        agents_map = self.agents_map()
        state = state * np.logical_not(
            agents_map > 0)
        state = state + agents_map
        return state

    def trash_by_segment(self, agent_id: str) -> int:
        """Calculate trash by segment

        Calculate number of trash items on a given
        segment

        Args:
            agent_id: agent ID in form `agent_{i}`

        Returns:
            int: number of trash items on a given segment
        """
        segment = self.trash_segment_map[agent_id]
        trash_by_segment = np.logical_and(
            segment, self.trash_grid
        )
        return trash_by_segment.sum()

    def resource_by_segment(self, agent_id: str) -> int:
        """Calculate resource by segment

        Calculate number of resource items on a given
        segment

        Args:
            agent_id: agent ID in form `agent_{i}`

        Returns:
            int: number of resource items on a given segment
        """
        segment = self.resource_segment_map[agent_id]
        resource_by_segment = np.logical_and(
            segment, self.resource_grid
        )
        return resource_by_segment.sum()

    def is_home_segment(self, agent_id) -> bool:
        """Check if home segment

        Check if given agent is on home segment

        Args:
            agent_id: agent ID in form `agent_{i}`

        Returns:
            bool: True if currently on home segment
        """
        loc = self.agents_state[agent_id]['loc']
        home = self.segment_map[agent_id][loc[0], loc[1]]
        return bool(home.item())

    def add_block(self, agent_id: str) -> None:
        """Add block

        Impose block on agent actions (resource and trash
        recycling)

        Args:
            agent_id: agent ID in form `agent_{i}`

        Returns:
            None
        """
        if not self.blocked[agent_id]:
            self.blocked[agent_id] = True
            self.blocked_map = np.logical_or(
                self.blocked_map, self.segment_map[agent_id])
            blocked_segment = np.logical_not(self.segment_map[agent_id])
            blocked_segment = blocked_segment + self.blocked_vanish_alpha
            blocked_segment = np.clip(blocked_segment, a_min=0, a_max=1)
            blocked_segment = np.expand_dims(blocked_segment, axis=-1)
            new_st = self.substrate_texture * blocked_segment
            new_st = np.round(new_st, 0).astype(np.uint8)
            self.substrate_texture = new_st

    def pad_state(self, state: np.ndarray) -> np.ndarray:
        """Pad state

        Pad image state for building local observations

        Args:
            state: raw engine visual state

        Returns:
            np.ndarray: padded engine visual state
        """
        p = self.obs_dim
        state_pad = np.pad(
            state, pad_width=((p, p), (p, p), (0, 0)),
            constant_values=0)
        return state_pad

    def delete_resource(self, center_loc: tuple) -> None:
        """Delete resource

        Delete resource from game grid given its raw
        numpy loc

        Args:
            center_loc: numpy loc on the 2D game field

        Returns:
            None
        """
        # delete from the references
        grid_loc = tuple(self.resource_center_to_grid[center_loc])
        del self.resource_center_to_grid[center_loc]
        del self.resource_grid_to_center[grid_loc]
        # delete from the grid
        self.resource_grid[grid_loc[0], grid_loc[1]] = 0
        # delete from the resource map
        y, x = center_loc
        c = (self.resource_size // 2)
        self.resource_map.paste(
            self.erasers['resource'], (x - c, y - c)
        )

    def delete_trash(self, center_loc: tuple) -> str:
        """Delete trash

        Delete trash from game grid given its raw
        numpy loc

        Args:
            center_loc: numpy loc on the 2D game field

        Returns:
            None
        """
        # delete from the references
        grid_loc = tuple(self.trash_center_to_grid[center_loc])
        del self.trash_center_to_grid[center_loc]
        del self.trash_grid_to_center[grid_loc]
        trash_source = self.trash_center_to_source[center_loc]
        del self.trash_center_to_source[center_loc]
        # delete from the grid
        self.trash_grid[grid_loc[0], grid_loc[1]] = 0
        # delete from the trash map
        y, x = center_loc
        c = (self.trash_size // 2)
        self.trash_map.paste(
            self.erasers['trash'], (x - c, y - c)
        )
        return trash_source

    def add_resource(self, grid_loc: np.ndarray) -> None:
        """Add resource

        Add resource to the game field and resource grid
        given its resource grid location

        Args:
            grid_loc: numpy loc resource grid

        Returns:
            None
        """
        center_loc = grid_loc * self.resource_size + self.resource_size // 2
        # add to the references
        self.resource_grid_to_center[tuple(grid_loc)] = center_loc
        self.resource_center_to_grid[tuple(center_loc)] = grid_loc
        # add to the resource grid
        self.resource_grid[grid_loc[0], grid_loc[1]] = 1
        # add to the resource map
        y, x = tuple(center_loc)
        c = (self.resource_size // 2)
        self.resource_map.paste(
            self.icons['resource'], (x - c, y - c)
        )

    def add_trash(self, grid_loc: np.ndarray, agent_id: str) -> None:
        """Add trash

        Add trash to the game field and trash grid
        given its trash grid location

        Args:
            grid_loc: numpy loc trash grid
            agent_id: agent ID in form `agent_{i}`

        Returns:
            None
        """
        center_loc = grid_loc * self.trash_size + self.trash_size // 2
        # add to the references
        self.trash_grid_to_center[tuple(grid_loc)] = center_loc
        self.trash_center_to_grid[tuple(center_loc)] = grid_loc
        self.trash_center_to_source[tuple(center_loc)] = agent_id
        # add to the trash grid
        self.trash_grid[grid_loc[0], grid_loc[1]] = 1
        # add to the trash map
        y, x = tuple(center_loc)
        c = (self.trash_size // 2)
        self.trash_map.paste(
            self.icons['trash'], (x - c, y - c)
        )

    def sample_trash(self, agent_id: str) -> bool:
        """Sample trash

        Sample trash randomly at a given agents' segment

        Args:
            agent_id: agent ID in form `agent_{i}`

        Returns:
            bool: True if sampled successfully and False otherwise
        """
        a_reg = np.logical_and(
            self.trash_segment_map[agent_id],
            np.logical_and(
                np.logical_not(self.non_trash_grid),
                np.logical_not(self.trash_grid)
            )
        ).astype(np.uint8)
        a_loc = np.argwhere(a_reg > 0)
        if a_loc.shape[0] > 0:
            idx = self.rng.integers(low=0, high=a_loc.shape[0])
            grid_loc = a_loc[idx]
            self.add_trash(grid_loc=grid_loc, agent_id=agent_id)
            done = True
        else:
            done = False
        return done

    def sample_resource(self, agent_id: str) -> bool:
        """Sample resource

        Sample resource randomly at a given agents' segment

        Args:
            agent_id: agent ID in form `agent_{i}`

        Returns:
            bool: True if sampled successfully and False otherwise
        """
        a_reg = np.logical_and(
            self.resource_segment_map[agent_id],
            np.logical_and(
                np.logical_not(self.non_resource_grid),
                np.logical_not(self.resource_grid)
            )
        ).astype(np.uint8)
        a_loc = np.argwhere(a_reg > 0)
        if a_loc.shape[0] > 0:
            idx = self.rng.integers(low=0, high=a_loc.shape[0])
            grid_loc = a_loc[idx]
            self.add_resource(grid_loc=grid_loc)
            done = True
        else:
            done = False
        return done

    def move_candidate(
        self, action_id: int,
        agent_id: str, step_size: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Movement candidate

        Propose movement candidate without taking into an account
        unreachable regions

        Args:
            action_id: movement action integer id
            agent_id: agent ID in form `agent_{i}`
            step_size: agent step size in pixels

        Returns:
            Tuple[np.ndarray, np.ndarray, float]: tuple which contains:
                - initial location
                - new candidate location
                - new direction
        """
        loc = self.agents_state[agent_id]['loc']
        dir = self.agents_state[agent_id]['dir_pi']
        if action_id == 0:
            new_dir = dir
        elif action_id == 1:
            new_dir = (dir + 0.5) % 2.
        elif action_id == 2:
            new_dir = (dir - 0.5) % 2.
        elif action_id == 3:
            new_dir = (dir - 1.) % 2.
        else:
            new_dir = None
            raise ValueError(
                f'Invalid action_id for movement: {action_id}'
            )
        shift = dir_to_shift(dir=new_dir)
        new_loc = loc + shift * step_size
        new_loc = np.clip(
            new_loc,
            a_min=(0 + self.agent_size // 2),
            a_max=(self.grid_size - 1 - self.agent_size // 2)
        ).astype(int)
        return loc, new_loc, new_dir

    def move(self, action_id: int, agent_id: str) -> None:
        """Make movement

        Make agent movement with respect to unreachable regions

        Args:
            action_id: movement action integer id
            agent_id: agent ID in form `agent_{i}`

        Returns:
            None
        """
        valid = False
        new_loc, new_dir = None, None
        step = self.move_step
        while not valid and step > -1:
            init_loc, new_loc, new_dir = self.move_candidate(
                action_id=action_id, agent_id=agent_id,
                step_size=step
            )
            valid = not bool(self.non_reachable[new_loc[0], new_loc[1]])
            step -= 1
        self.agents_state[agent_id]['loc'] = new_loc
        self.agents_state[agent_id]['dir_pi'] = new_dir

    def pickup(
        self, agent_id: str, type: str
    ) -> Tuple[np.ndarray, bool]:
        """Pickup item

        pickup item from game field

        Args:
            agent_id: agent ID in form `agent_{i}`
            type: one of {'resource', 'trash'}

        Returns:
            Tuple[np.ndarray, bool]: picked item loc and pickup
                status boolean
        """
        status, item_loc = False, None
        loc = self.agents_state[agent_id]['loc']
        reach = self.agent_reach
        if type == 'resource':
            ridxs = np.array(list(self.resource_grid_to_center.values()))
        elif type == 'trash':
            ridxs = np.array(list(self.trash_grid_to_center.values()))
        else:
            raise ValueError(
                f'Invalid item to pickup: {type}'
            )
        if ridxs.shape[0] > 0:
            deltas = np.abs(ridxs - loc)
            max_dist = deltas.max(axis=1)
            valid = np.argwhere(max_dist < reach).squeeze(axis=-1)
            if len(valid) > 0:
                if len(valid) > 1:
                    mean_dist = deltas[valid].mean(axis=-1)
                    min_dist = np.argmin(mean_dist)
                    valid = valid[min_dist]
                else:
                    valid = valid[0]
                item_loc = ridxs[valid]
                if not self.agents_state[agent_id]['inventory'][type]:
                    if type == 'resource':
                        self.delete_resource(center_loc=tuple(item_loc))
                    else:
                        trash_source = self.delete_trash(center_loc=tuple(item_loc))
                        self.agents_state[agent_id]['inventory']['trash_source'] = trash_source
                    self.agents_state[agent_id]['inventory'][type] = True
                    status = True
        return item_loc, status

    def throw_resource(self, agent_id: str) -> bool:
        """Throw resource

        Throw resource onto the game field at the current
        agent location if possible

        Args:
            agent_id: agent ID in form `agent_{i}`

        Returns:
            bool: True if succeeded, False otherwise
        """
        ac_loc = self.agents_state[agent_id]['loc']
        ag_loc = ac_loc // self.resource_size
        done = False
        if self.agents_state[agent_id]['inventory']['resource']:
            # check if we can throw it here
            invalid = np.logical_or(
                self.non_resource_grid,
                self.resource_grid
            )
            valid = not bool(invalid[ag_loc[0], ag_loc[1]])
            if valid:
                self.add_resource(grid_loc=ag_loc)
                # remove from inventory
                self.agents_state[agent_id]['inventory']['resource'] = False
                done = True
        return done

    def throw_trash(self, agent_id: str) -> bool:
        """Throw trash

        Throw trash onto the game field at the current
        agent location if possible

        Args:
            agent_id: agent ID in form `agent_{i}`

        Returns:
            bool: True if succeeded, False otherwise
        """
        ac_loc = self.agents_state[agent_id]['loc']
        ag_loc = ac_loc // self.trash_size
        done = False
        if self.agents_state[agent_id]['inventory']['trash']:
            # check if we can throw it here
            invalid = np.logical_or(
                self.non_trash_grid,
                self.trash_grid
            )
            valid = not bool(invalid[ag_loc[0], ag_loc[1]])
            if valid:
                trash_source = self.agents_state[agent_id]['inventory']['trash_source']
                self.add_trash(grid_loc=ag_loc, agent_id=trash_source)
                # remove from inventory
                self.agents_state[agent_id]['inventory']['trash'] = False
                self.agents_state[agent_id]['inventory']['trash_source'] = None
                done = True
        return done

    def recycle_resource(self, agent_id: str) -> bool:
        """Recycle resource

        Recycle resource at the fabricator if:
            1) It is in the inventory
            2) Fabricator is close enough
            3) Agent is not blocked

        Args:
            agent_id: agent ID in form `agent_{i}`

        Returns:
            bool: True if succeeded, False otherwise
        """
        done = False
        if not self.blocked[agent_id]:
            if self.agents_state[agent_id]['inventory']['resource']:
                ac_loc = self.agents_state[agent_id]['loc']
                machine_loc = self.machines[agent_id]['center']
                delta = np.abs(machine_loc - ac_loc).max()
                if delta <= self.machine_reach:
                    self.agents_state[agent_id]['inventory']['resource'] = False
                    self.agents_state[agent_id]['wealth'] += self.resource_price
                    self.sample_trash(agent_id=agent_id)
                    done = True
        return done

    def recycle_trash(self, agent_id: str) -> bool:
        """Recycle trash

        Recycle trash at the recycler if:
            1) It is in the inventory
            2) Recycler is close enough
            3) Agent is not blocked
            4) Agent has enough money for recycling

        Args:
            agent_id: agent ID in form `agent_{i}`

        Returns:
            bool: True if succeeded, False otherwise
        """
        done = False
        if not self.blocked[agent_id]:
            has_trash = self.agents_state[agent_id]['inventory']['trash']
            has_money = self.agents_state[agent_id]['wealth'] >= self.recycle_cost
            if has_trash and has_money:
                ac_loc = self.agents_state[agent_id]['loc']
                recycler_loc = self.recyclers[agent_id]['center']
                delta = np.abs(recycler_loc - ac_loc).max()
                if delta <= self.machine_reach:
                    self.agents_state[agent_id]['inventory']['trash'] = False
                    self.agents_state[agent_id]['inventory']['trash_source'] = None
                    self.agents_state[agent_id]['wealth'] -= self.recycle_cost
                    done = True
        return done

    def drop_resource(self, agent_id: str) -> str:
        """Drop resource

        Throw or recycle resource. If recycle is possible,
        recycle it at the fabricator, otherwise drop onto
        the game field

        Args:
            agent_id: agent ID in form `agent_{i}`

        Returns:
            str: drop status string. One in
                {'dropped', 'drop_failed', 'resource_recycled'}
        """
        recycle_done = self.recycle_resource(agent_id=agent_id)
        if not recycle_done:
            throw_done = self.throw_resource(agent_id=agent_id)
            if throw_done:
                status = 'dropped'
            else:
                status = 'drop_failed'
        else:
            status = 'resource_recycled'
        return status

    def drop_trash(self, agent_id: str) -> str:
        """Drop trash

        Throw or recycle trash. If recycle is possible,
        recycle it at the recycler, otherwise drop onto
        the game field

        Args:
            agent_id: agent ID in form `agent_{i}`

        Returns:
            str: drop status string. One in
                {'dropped', 'drop_failed', 'resource_recycled'}
        """
        recycle_done = self.recycle_trash(agent_id=agent_id)
        if not recycle_done:
            throw_done = self.throw_trash(agent_id=agent_id)
            if throw_done:
                status = 'dropped'
            else:
                status = 'drop_failed'
        else:
            status = 'trash_recycled'
        return status

    def local_view(self, state_pad: np.ndarray, agent_id: str) -> np.ndarray:
        """Render local view

        Render local view image array for a given agent ID

        Args:
            state_pad: padded engine state image
            agent_id: agent ID in form `agent_{i}`

        Returns:
            np.ndarray: local view image array
        """
        agent_loc = self.agents_state[agent_id]['loc']
        agent_dir = self.agents_state[agent_id]['dir_pi']
        p = self.obs_dim
        if agent_dir == 0.0:
            x_high = agent_loc[0] + p // 2  # right
            x_low = x_high - p  # left
            y_low = agent_loc[1] - self.agent_size // 2 + 1  # back
            y_high = y_low + p  # forward
        elif agent_dir == 0.5:
            x_high = agent_loc[0] + self.agent_size // 2  # back
            x_low = x_high - p  # forward
            y_high = agent_loc[1] + p // 2  # right
            y_low = y_high - p  # left
        elif agent_dir == 1.0:
            x_low = agent_loc[0] - p // 2 + 1  # right
            x_high = x_low + p  # left
            y_high = agent_loc[1] + self.agent_size // 2  # back
            y_low = y_high - p  # forward
        elif agent_dir == 1.5:
            x_low = agent_loc[0] - self.agent_size // 2 + 1  # back
            x_high = x_low + p  # forward
            y_low = agent_loc[1] - p // 2 + 1  # right
            y_high = y_low + p  # left
        else:
            raise ValueError(
                f'Invalid agent direction: {agent_dir}'
            )
        xl, xh, yl, yh = x_low + p, x_high + p, y_low + p, y_high + p
        obs = state_pad[xl:xh, yl:yh, :]
        all_machines = self.all_machines[xl:xh, yl:yh, :]
        perspective = self.agents_perspectives[agent_id][xl:xh, yl:yh, :]
        obs = obs * np.logical_not(all_machines)
        obs = obs + perspective
        if agent_dir == 0.0:
            obs = np.rot90(obs, k=1)
        elif agent_dir == 0.5:
            pass
        elif agent_dir == 1.0:
            obs = np.rot90(obs, k=-1)
        elif agent_dir == 1.5:
            obs = np.rot90(obs, k=2)
        else:
            pass
        return obs.copy()

    def local_proprio(self, agent_id: str) -> np.ndarray:
        """Get local proprioceptive obs

        Get local proprioceptive obs for a given agent ID

        Args:
            agent_id: agent ID in form `agent_{i}`

        Returns:
            np.ndarray: subjective proprioceptive obs
        """
        money = self.agents_state[agent_id]['wealth'] / self.resource_price
        dir = self.agents_state[agent_id]['dir_pi']
        loc = self.agents_state[agent_id]['loc']
        center = self.centers[agent_id]

        # North
        if dir == 0.0:
            x, y = 1, 0
        elif dir == 0.5:
            x, y = 0, 1
        elif dir == 1.0:
            x, y = -1, 0
        elif dir == 1.5:
            x, y = 0, -1
        else:
            x, y = 0, 0

        center_vec = loc_to_coord(center, self.grid_size) - \
            loc_to_coord(loc, self.grid_size)
        rho, phi = cart2pol(center_vec[0].item(), center_vec[1].item())
        rho, phi = rho / self.grid_size, phi / np.pi
        is_home = float(self.is_home_segment(agent_id=agent_id))
        has_resource = float(
            self.agents_state[agent_id]['inventory']['resource'])
        has_trash = float(self.agents_state[agent_id]['inventory']['trash'])
        _, north_dir = cart2pol(x, y)
        north_dir = north_dir / np.pi
        proprio = np.array([
            money, has_resource, has_trash,
            rho, phi, north_dir, is_home
        ])
        return proprio.astype(np.float32)

    def local_obs(
        self, state_pad: np.ndarray, agent_id: str
    ) -> Dict[str, np.ndarray]:
        """Get local observation

        Get local composite observation for a given agent ID

        Args:
            state_pad: padded engine state image
            agent_id: agent ID in form `agent_{i}`

        Returns:
            Dict[str, np.ndarray]: composite observation with the
                following key-value pairs:
                    - 'image': local visual observation
                    - 'proprio': subjective proprioceptive observation
        """
        view = self.local_view(state_pad=state_pad, agent_id=agent_id)
        proprio = self.local_proprio(agent_id=agent_id)
        return {'image': view, 'proprio': proprio}
