# TraPSNet
- the requirements to use the experiments are
1. python3
2. tensorflow
3. unittest
4. multiprocessing
5. threading
6. shutil
7. better_exceptions
8. pickle
9. networkx
10. scipy

- before starting the experiments run these commands from the main folder

```sh
for i in {1..30}
do
cp ./gym/envs/rddl/rddl/lib/clibxx.so ./gym/envs/rddl/rddl/lib/clibxx$i.so
cp ./rddl/lib/clibxx.so ./rddl/lib/clibxx$i.so
done
```
- please see the individual folder on how to run the experiments
- if you encounter C/C++ errors, please email