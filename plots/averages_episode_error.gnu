#
# This script demonstrates the use of assignment operators and
# sequential expression evaluation to track data points as they
# are read in.
#
# We use the '=' and ',' operators to track the running total
# and previous 5 values of a stream of input data points.
#
# Ethan A Merritt - August 2007
#
# Define a function to calculate average over previous 5 points
#
set key invert box center right reverse Left
set xtics nomirror
set ytics nomirror
set border 3

samples(x) = $0 > 4 ? 5 : ($0+1)
avg5(x) = (shift5(x), (back1+back2+back3+back4+back5)/samples($0))
shift5(x) = (back5 = back4, back4 = back3, back3 = back2, back2 = back1, back1 = x)

# Initialize a running sum
init(x) = (back1 = back2 = back3 = back4 = back5 = sum = 0)

#
# Plot data, running average and cumulative average
#

datafile = 'data/episode_error_plot.dat'
#set xrange [0:57]

set style data linespoints

plot sum = init(0), \
     datafile using 0:1 title 'data' lw 1 lc rgb 'forest-green', \
     '' using 0:(avg5($1)) title "running mean over previous 5 points" ps 0.5 lw 1 lc rgb "blue", \
     '' using 0:(sum = sum + $1, sum/($0+1)) title "cumulative mean" lw 1 lc rgb "dark-red"
