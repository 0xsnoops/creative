def priority(item, capacities):
    # Earliest-opened bin that fits wins; the new bin is last, so it is chosen only if nothing fits.
    return [-i for i in range(len(capacities))]
