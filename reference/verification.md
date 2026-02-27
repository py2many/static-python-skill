## Verification
smt.pre and smt.post are transpiled to z3 to verify correctness. The code is ignored at runtime.

uvx py2many --smt test.py - | z3 -smt2 -in

if UNSAT and there is a counter example, fix it first before continuing.