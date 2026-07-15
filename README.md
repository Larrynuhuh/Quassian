Quassian is a very simplified implementation of Alex Townsend and Nicholas Hale's paper on Fast and Accurate Computation of
Gauss Legendre in JAX. It is more useful in cases where the function being integrated is rather not too complex and has a smooth graph or anything of such matters due to it being a toy model, because high amounts of precision can be calculated by taking higher 'n'. 
The project is fundamentally revolving around asymptotic formulae, due to this, it'd be unwise to take 'n' below 100 for this.

Expect the program to lose enormous amounts of accuracy if incredibly low 'n' are selected, and the program is bound to break
at 'n' <= 10, do not try it. Besides, when 'n' is so low, it'd be better to simply use scipy for this, which provides a 
surprising amount of speed for such low 'n' and also is quite good and rather fast at n = 100, though its implementation of the
GW method obviously starts lacking when we start to increase 'n' to incredibly high degrees such as 10,000 or more. 

The implementation is also rather slow, perhaps due to jax.pure_callback calls. Overall, this was quite a complex project
so I couldn't really optimize it further without losing my mind. Though this project cannot achieve high precision 
as promised in the Townsend-Hale paper, for rather well-behaved functions with smooth graphs with a generous amount of 'n'
we can achieve a stable 1e-6 to 1e-10 error in the total integration. You may, however, find compute_nodes a far more 
useful thing, allowing extremely fast computation of the nodes of the legendre polynomials with errors getting sparser and 
lower as we approach infinity. 

