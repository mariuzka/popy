"""Various utility functions for popy."""
from __future__ import annotations

import typing

from bokehgraph import BokehGraph
import networkx as nx
from networkx import bipartite
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

import popy

if typing.TYPE_CHECKING:
    from popy import AgentList



##########################################################################
# scientific stuff
##########################################################################

def create_agent_graph(
        agents: AgentList,
        node_attrs: list | None = None,
        include_0_weights: bool = True,
) -> nx.Graph:
    """Create a Graph from a model's agent list.

    Args:
        agents: A model's agent list
        node_attrs: A list of agent attributes
        include_0_weights: Should edges with weight 0 be displayed?

    Returns:
        A weighted graph created from a model's agent list. Agents are connected if they are
        neighbors in the model. Their connecting edge include the contact_weight as "weight"
        attribute.
    """
    projection = nx.Graph()

    # create nodes
    for agent in agents:
        if not projection.has_node(agent.id):

            node_attr_dict = {}
            if node_attrs is not None:
                for node_attr in node_attrs:
                    node_attr_dict.update({node_attr: vars(agent)[node_attr]})
            projection.add_node(agent.id, **node_attr_dict)

    # create edges
    for agent in agents:
        for agent_v in agent.neighbors():
            if not projection.has_edge(agent.id, agent_v.id):
                weight = agent.contact_weight(agent_v)
                if include_0_weights or weight > 0:
                    projection.add_edge(agent.id, agent_v.id, weight=weight)

    return projection


def export_network(env) -> nx.Graph:
    """Export the current agent network (unweighted version).

    This is a projection of the underlying bipartite network between agents and locations.

    Returns:
        The current agent network as unipartite, unweighted graph.
    """
    agent_nodes = {n for n, d in env.g.nodes(data=True) if d["bipartite"] == 0}

    projection = bipartite.projection.projected_graph(env.g, agent_nodes)
    for agent_id in projection:
        del projection.nodes[agent_id]["_obj"]

    return projection


# TODO: calculate relative freqs
def create_contact_matrix(
    agents: list | AgentList,
    attr: str = "id",
    weighted: bool = False,
    plot: bool = False,
    annot: bool = False,
) -> pd.DataFrame:
    """Create a contact matrix as a DataFrame from a given model's agent list.

    Args:
        agents: A list of agents.
        attr: The agent attribute which is shown in the matrix.
        weighted: Should the contacts be weighted? Defaults to False.
        plot: Should the matrix be plotted? Defaults to False.
        annot: Should the plottet matrix be annotated? Defaults to False.

    Returns:
        A DataFrame containing a contact matrix based on `attr`.
    """
    contact_data = []
    attr_values = []
    pairs = []

    attr_u_name = f"{attr}_u"
    attr_v_name = f"{attr}_v"

    for agent_u in agents:
        attr_u = getattr(agent_u, attr)
        attr_values.append(attr_u)

        for agent_v in agent_u.neighbors():
            attr_v = getattr(agent_v, attr)
            attr_values.append(attr_v)

            pair = {agent_u.id, agent_v.id}

            if pair not in pairs:
                contact_data.append(
                    {
                        "id_u": agent_u.id,
                        attr_u_name: attr_u,
                        "id_v": agent_v.id,
                        attr_v_name: attr_v,
                        "weight": agent_u.contact_weight(agent_v),
                    },
                )
                pairs.append(pair)

    attr_values = list(set(attr_values))
    df = pd.DataFrame(index=sorted(attr_values, reverse=True), columns=sorted(attr_values))
    df = df.fillna(0)

    weight_total = 0
    for contact in contact_data:
        weight = contact["weight"] if weighted else 1
        weight_total += weight

        df.loc[contact[attr_u_name], contact[attr_v_name]] = (
            df.loc[contact[attr_u_name], contact[attr_v_name]] + weight
        )
        df.loc[contact[attr_v_name], contact[attr_u_name]] = (
            df.loc[contact[attr_v_name], contact[attr_u_name]] + weight
        )

    #df = df / 2

    if plot:
        g = sns.heatmap(df, annot=annot)
        g.set(xlabel=attr, ylabel=attr)

    return df


def eval_affiliations(agents, locations) -> None:
    """Prints information on the distribution of agents per location and locations per agent.

    Raises:
        PopyException: _description_
        PopyException: _description_
    """
    #if self.agents is None:
    #    msg = "You have to create agents first!"
    #    raise PopyException(msg)

    #if self.locations is None:
    #    msg = "You have to create locations first!"
    #    raise PopyException(msg)

    df_locations = pd.DataFrame(
        [
            {
                "location_class": str(type(location)).split(".")[-1].split("'")[0],
                "n_agents": len(location.agents),
            }
            for location in locations
        ],
    )

    print_header("Number of agents per location")
    print(df_locations.groupby("location_class").describe())

    df_agents = pd.DataFrame(
        [
            {
                "agent_id": agent.id,
                "n_affiliated_locations": len(agent.locations),
            }
            for agent in agents
        ],
    )

    print_header("Number of affiliated locations per agent")
    print(df_agents.n_affiliated_locations.describe())


def plot_network(
        agents: list | AgentList,
        node_color: str = "red",
        node_attrs: list | None = None,
        edge_alpha: str = "weight",
        edge_color: str = "black",
        include_0_weights: bool = True,
):
    """_summary_.

    Args:
        agents (list | AgentList): _description_
        node_color (str, optional): _description_. Defaults to "red".
        node_attrs (list | None, optional): _description_. Defaults to None.
        edge_alpha (str, optional): _description_. Defaults to "weight".
        edge_color (str, optional): _description_. Defaults to "black".
        include_0_weights (bool, optional): _description_. Defaults to True.
    """
    graph = create_agent_graph(agents, node_attrs=node_attrs, include_0_weights=include_0_weights)
    graph_layout = nx.drawing.spring_layout(graph)
    plot = BokehGraph(graph, width=500, height=500, hover_edges=True)
    plot.layout(layout=graph_layout)
    plot.draw(
        node_color=node_color,
        edge_alpha=edge_alpha,
        edge_color=edge_color,
    )


def location_information(
        locations: list[popy.Location],
        select_locations: popy.Location | list[popy.Location] | None = None,
        agent_attributes: str | None | list[str] | None = None,
        output_format:str = "table",
) -> None | pd.DataFrame:
    """Parameter
    - locations =>  Liste aller location Instanzen des Netzwerkes (streng genommen reicht auch Liste von allen die man sich anschauen will)
    - select_locations => einzelner Konstruktor oder Liste von Klassen Konstruktoren
    - agent_attributes => Liste oder einzelnenes Attribute der Agenten, die in der Output-Tabelle erscheinen sollen.

    Output: Pro Location Instance der ausgewählten Location Klassen werden
            Pandas Dataframes (Columns gefiltert durch agent_attributes)
            mit Hilfe von tabulate als Konsolen print ausgegeben

    """
    if select_locations:
        if not isinstance(select_locations, list):
            select_locations = [select_locations]

    if agent_attributes:
        if not isinstance(agent_attributes, list):
            agent_attributes = [agent_attributes]

    # determine eligible locations classes
    valid_locations = []
    if select_locations:
        for location_instance in locations:

            for locationtype in select_locations:
                if isinstance(location_instance, locationtype):
                    valid_locations.append(location_instance)
    else:
        valid_locations = list(locations)


    # create agent df per location instance
    agent_dfs = {}
    if agent_attributes:

        for i, location_instance in enumerate(valid_locations):

            # Create the title of printout
            title = f'{i+1}.Location: {str(location_instance).split(" ")[0]}'
            location_type = str(location_instance).split(" ")[0]
            # get all agents per location instance, subset df by agent-attributes
            df = get_df_agents(agents = location_instance.agents)
            df = df[list(agent_attributes)]
            df["location_type"] = location_type
            agent_dfs[title] = df
    else:

        for i, location_instance in enumerate(valid_locations):
            title = f'{i+1}.Location: {str(location_instance).split(" ")[0]}'
            location_type = str(location_instance).split(" ")[0]
            df = get_df_agents(agents = location_instance.agents)
            df.drop(df.iloc[:,0:7], axis = 1, inplace = True)
            df["location_type"] = location_type
            agent_dfs[title] = df

    if output_format == "table":
    #### Print Part "Basic"
        for  title, df in agent_dfs.items():
            print(f"{title} \n")
            print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
            print("\n")

    if output_format == "df":
        location_id_counter = 0
        df_list = []
        for title, df in agent_dfs.items():
            df.insert(0, "location_id", [location_id_counter]*len(df.index))
            location_id_counter += 1
            df_list.append(df)
        return pd.concat(df_list, ignore_index=True)
    return None





def location_crosstab(
        locations: list[popy.Location],
        select_locations: popy.Location | list[popy.Location],
        agent_attributes: str | list[str],
        output_format = "table",
)-> list[pd.DataFrame] | None:
    # Make every Parameter a list
    if select_locations:
        if not isinstance(select_locations, list):
            select_locations = [select_locations]

    if agent_attributes:
        if not isinstance(agent_attributes, list):
            agent_attributes = [agent_attributes]

    # determine eligible locations classes
    valid_locations = []
    if select_locations:
        for location_instance in locations:

            for locationtype in select_locations:
                if isinstance(location_instance, locationtype):
                    valid_locations.append(location_instance)
    else:
        valid_locations = list(locations)


    # create agent df per location instance
    if output_format == "table":
        agent_dfs = {}
        if agent_attributes:

            for i, location_instance in enumerate(valid_locations):

                title = f'{i+1}.Location: {str(location_instance).split(" ")[0]}'
                df = get_df_agents(agents=location_instance.agents)
                df = df[list(agent_attributes)]
                agent_dfs[title] = df

        for title, df in agent_dfs.items():

            for agent_attribute in agent_attributes:

                crosstab_table = pd.crosstab(
                    index= df[agent_attribute],
                    columns="count",
                )

                print(f"{title}")
                print(
                    tabulate(
                        crosstab_table,
                        headers=[agent_attribute, "count"],
                        tablefmt="fancy_grid",
                    ),
                )
                print("\n")
    if output_format == "df":
        agent_dfs = {}
        result_list = []
        for i, location_instance in enumerate(valid_locations):
            title = f'{i+1}.Location: {str(location_instance).split(" ")[0]}'
            location_type = str(location_instance).split(" ")[0]
            df = get_df_agents(agents=location_instance.agents)

            # only keep wanted columns (agent attributes)
            df = df[list(agent_attributes)]

            # add location type as str
            df["location_type"] = location_type

            agent_dfs[title] = df


        for agent_attribute in agent_attributes:

            location_id = 0
            df_list_of_attribute = []
            for title, df in agent_dfs.items():

                crosstab_table = pd.crosstab(
                    index= df[agent_attribute],
                    columns="count",
                )
                crosstab_table.insert(0, "location_id", [location_id]*len(crosstab_table.index))
                crosstab_table["location_type"] = df["location_type"][0]

                df_list_of_attribute.append(crosstab_table)

                location_id += 1
            df_attribute = pd.concat(df_list_of_attribute, ignore_index=False)

            # fix columns after concat
            df_attribute.reset_index(inplace=True)
            df_attribute.columns.names = ["index"]
            location_array = df_attribute["location_id"].copy()
            df_attribute.drop(columns="location_id", inplace=True)
            df_attribute.insert(0, "location_id", location_array)

            result_list.append(df_attribute)
        return result_list
    return None


##########################################################################
# technical helper functions
##########################################################################


def get_df_agents(agents: AgentList) -> pd.DataFrame:
    """_summary_.

    Args:
        agents (AgentList): _description_

    Returns:
        pd.DataFrame: _description_
    """
    return pd.DataFrame([vars(agent) for agent in agents])


def group_it(
    value: int | float,
    start: int | float,
    step: int,
    n_steps: int,
    return_value: typing.Literal["index", "lower_bound", "range"] = "index",
    summarize_highest: bool = False,
) -> int | float | tuple[int | float, int | float]:
    """_summary_.

    Args:
        value: _description_
        start: _description_
        step: _description_
        n_steps: _description_
        return_value: _description_. Defaults to "index".
        summarize_highest: _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # TODO: something is fishy in this function...
    assert type(value) in [int, float], f"{value} has to be a number!"
    assert value >= start, f"The value {value} is smaller than the smallest lower bound {start}."

    for i in range(n_steps):
        lower_bound = start + step * i
        upper_bound = lower_bound + step

        if lower_bound <= value:
            if return_value == "index":
                new_value = i

            elif return_value == "lower_bound":
                new_value = lower_bound  # type: ignore

            elif return_value == "range":
                new_value = (lower_bound, upper_bound)  # type: ignore

            else:
                msg = "You have entered a non-existing option for `return_value`."
                raise Exception(msg)


        if not summarize_highest:
            if i == n_steps + 1:
                if value > upper_bound:
                    new_value = np.nan
    # BUG: new_value possibly unbound
    return new_value

def print_header(text: object):
    """Print a header around an object.

    Args:
        text: Object to be printed within the header.
    """
    # TODO: This is weird.... and it should probably not be print-statements cuz no one can catch
    # them.
    print("")
    print("")
    print("______________________________________")
    print(text)
    print("______________________________________")
    print("")


def network_measures(agent_list, node_attrs = None) -> dict | list[dict]:
    """Creates nx networkgraph and calculates common network measures.

    If the created network consist of independent groups of nodes
    subgraphs are created and measures are calculated for each subgraph

    Args:
        agent_list: A model's agent list
        node_attrs: A list of agent attributes

    Return:
        dictionary/or list of dictionaries of the common network
        measure results
    """
    result_dict = {}
    nx_graph = create_agent_graph(agent_list, node_attrs)

    # make distinction between multiple independent networks and one network
    if nx.is_connected(nx_graph):

        result_dict["diameter"] = nx.diameter(nx_graph, weight = "weight")
        result_dict["density"] = nx.density(nx_graph)
        result_dict["transitivity"] = nx.transitivity(nx_graph)
        result_dict["avg_clustering"] = nx.average_clustering(
            nx_graph,
            weight= "weight",
        )
        result_dict["avg_path_length"] = nx.average_shortest_path_length(
            nx_graph,
            weight = "weight",
        )
        #result_dict["periphery"] = nx.periphery(nx_graph, weight = "weight")
        #result_dict["center"] = nx.center(nx_graph, weight = "weight")
        #result_dict["centrality"] = nx.degree_centrality(nx_graph)
        return result_dict
    else:
        # sort subgraph component size(=num of nodes) in ascending order
        component_list = sorted(
            nx.connected_components(nx_graph),key=len,
            reverse=False,
        )

        # create graph for each component and calculate network measures
        result_list = []
        for component in component_list:
            result_dict_subgraph = {}
            try:
                nx_subgraph = nx_graph.subgraph(component)
            except nx.NetworkXError:
                print("Cant make graph out of component")
                break

            result_dict_subgraph["diameter"] = nx.diameter(
                nx_subgraph,
                weight = "weight",
            )
            result_dict_subgraph["density"] = nx.density(nx_subgraph)
            result_dict_subgraph["transitivity"] = nx.transitivity(nx_subgraph)
            result_dict_subgraph["avg_clustering"] = nx.average_clustering(
                nx_subgraph,
                weight= "weight",
            )
            result_dict_subgraph["avg_path_length"] = nx.average_shortest_path_length(
                nx_subgraph,
                weight = "weight",
            )

            #result_dict_subgraph["centrality"] = nx.degree_centrality(nx_subgraph)
            #result_dict_subgraph["periphery"] = nx.periphery(nx_subgraph, weight = "weight")
            #result_dict_subgraph["center"] = nx.center(nx_subgraph, weight = "weight")
            result_list.append(result_dict_subgraph)
        return result_list
