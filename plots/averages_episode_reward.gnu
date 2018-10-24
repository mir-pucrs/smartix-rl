set key invert box bottom right reverse

set xtics nomirror
set ytics nomirror

set grid ytics lc rgb "#bbbbbb" lw 1 lt 0
set grid xtics lc rgb "#bbbbbb" lw 1 lt 0

set border 2

set style line 1 linecolor rgb 'blue' linetype 1 linewidth 1
set style line 2 linecolor rgb 'red' linetype 1 linewidth 1
set style line 3 linecolor rgb 'orange' linetype 1 linewidth 1

samples(x) = $0 > 4 ? 5 : ($0+1)
avg5(x) = (shift5(x), (back1+back2+back3+back4+back5)/samples($0))
shift5(x) = (back5 = back4, back4 = back3, back3 = back2, back2 = back1, back1 = x)
# Initialize a running sum
init(x) = (back1 = back2 = back3 = back4 = back5 = sum = 0)

datafile = 'data/episode_reward_plot.dat'

plot sum = init(0), \
     datafile using 0:1 title 'data' with lines linestyle 1, \
     '' using 0:(sum = sum + $1, sum/($0+1)) title "cumulative mean" with lines linestyle 2,  \
     '' using 0:(avg5($1)) title "running mean over previous 5 points" with lines linestyle 3