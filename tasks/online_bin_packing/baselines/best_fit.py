def priority(item, capacities):
    # Tightest fit wins (least space left over); ties go to the earliest bin.
    return [-(c - item) for c in capacities]
