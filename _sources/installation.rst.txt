.. currentmodule:: popy
.. highlight:: shell

============
Installation
============

To install the latest release of popy,
run the following command on your console:

.. code-block:: console

	$ pip install popy

Dependencies
------------

Popy supports Python 3.8 and higher.
The installation includes the following packages:

- `agentpy <https://agentpy.readthedocs.io/en/latest/>`_ as a modeling framework
- `numpy <https://numpy.org>`_ for scientific computing
- `matplotlib <https://matplotlib.org/>`_, for visualization
- `pandas <https://pandas.pydata.org>`_, for data manipulation
- `networkx <https://networkx.org/documentation/>`_, for networks/graphs
- `SALib <https://salib.readthedocs.io/>`_, for sensitivity analysis
- `joblib <https://joblib.readthedocs.io/>`_, for parallel processing

These optional packages can further be useful in combination with popy:

- `jupyter <https://jupyter.org/>`_, for interactive computing
- `ipysimulate <https://ipysimulate.readthedocs.io/>`_ >= 0.2.0, for interactive simulations
- `ema_workbench <https://emaworkbench.readthedocs.io/>`_, for exploratory modeling
- `seaborn <https://seaborn.pydata.org/>`_, for statistical data visualization

Development
-----------

The most recent version of agentpy can be cloned from Github:

.. code-block:: console

	$ git clone https://github.com/mariuzka/popy.git

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install -e


.. _Github repository: https://github.com/mariuzka/popy
