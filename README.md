# Newton
The first co-trained PINN problems: a block and a rocket block

I am still learning how to present the problem here in a readable manner.

### The Block Problem
A block of mass 1kg is at rest on a horizontal, frictionless surface, with no force applied to it. At some time we’ll call t = 0, a force begins being applied to the block. Determine the force as a function of time that will cause the block to be at specific locations at specific times, under the constraint that the jerk (d<sup>3</sup>x/dt<sup>3</sup> = dF/dt /mass) may not be greater than 40m/s<sup>3</sup>.
<table>
<thead>
<tr>
<th>Time</th>
<th>Position</th>
</tr>
</thead>
<tbody>
<tr>
<td>0 s</td>
<td>0 m</td>
</tr>
<tr>
<td>1 s</td>
<td>4 m</td>
</tr>
<tr>
<td>4 s</td>
<td>5 m</td>
</tr>
</tbody>
</table>

_Solved in newton_v1.py_


### The Rocket Block Problem
Now consider that the block is actually a rocket block that is burning its own mass to provide that force. The rate of change of mass is dm/dt = - a abs(F). Find the function F(t) that minimizes the burned mass while adhering to all other constraints. a = 1e-2; the block has 100Ns of impulse.

_Solved in newton_v2.py_
