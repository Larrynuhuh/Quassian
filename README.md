Quassian is a slightly simplified implementation of Alex Townsend and Nicholas Hale's paper on Fast and Accurate Computation of
Gauss Legendre in JAX. It is more useful in cases where the function being integrated is rather complex and has a very high 
frequency graph or anything of such matters, because high amounts of precision can be calculated by taking higher 'n'. 
The project is fundamentally revolving around asymptotic formulae, due to this, it'd be unwise to take 'n' below 100 for this.

Expect the program to lose enormous amounts of accuracy if incredibly low 'n' are selected, and the program is bound to break
at 'n' <= 10, do not try it. Besides, when 'n' is so low, it'd be better to simply use scipy for this, which provides a 
surprising amount of speed for such low 'n' and also is quite good and rather fast at n = 100, though its implementation of the
GW method obviously starts lacking when we start to increase 'n' to incredibly high degrees such as 10,000 or more. 

