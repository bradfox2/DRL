import pstats
from pstats import SortKey

p = pstats.Stats('profile.profile')
p.sort_stats('name')
p.sort_stats('cumulative').print_stats(10)