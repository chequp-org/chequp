Testing the code
------------------

Setup environment and compile CHEQUP as described . For the following, we assume conda was used.

.. code-block:: sh

    conda activate chequp
    conda install -y -c conda-forge pytest
    cd tests
    py.test

Add a new test
~~~~~~~~~~~~~~~

* In the ``/test``, folder create a new file, for example ``test_<test name>.py``, and add a new ``test_<test name>`` function similar to the existing ones (see ``test_1d.py`` for an example).

* Make a new file at ``tests/checksum/benchmarks_json/<test name>.json`` containing ``{}``.

* Run all tests using ``py.test``. The checksum of the new test should fail and print the new json file to the console.

* Copy the json from the console output into the ``<test name>.json`` file.

* Verify that ``py.test`` now passes.
