import unittest
from unittest.mock import patch

import networkx as nx

import community_detection_utils as cdu


class SweepPreparationReuseTests(unittest.TestCase):
    @staticmethod
    def make_graph() -> nx.Graph:
        """Return a labeled graph fixture for sweep tests."""
        graph = nx.karate_club_graph()
        nx.set_node_attributes(
            graph,
            {node: data["club"] for node, data in graph.nodes(data=True)},
            "label_gt",
        )
        return graph

    @staticmethod
    def fake_orc_flow(graph: nx.Graph, **kwargs: object) -> nx.Graph:
        """Return a copy with deterministic ORC values instead of running flow.

        Args:
            graph: GCC input whose topology and attributes must be preserved.
            **kwargs: ORC options accepted by the production flow function.

        Returns:
            A graph copy with deterministic ``ricciCurvature`` edge values,
            enabling fast tests of cutoff reuse without invoking the solver.
        """
        output = graph.copy()
        for index, (_, _, data) in enumerate(output.edges(data=True)):
            data["ricciCurvature"] = float(index)
        return output

    def test_compute_lrc_uses_cached_neighborhoods_without_changing_values(
        self,
    ) -> None:
        """Check LRC values for triangles, paths, cycles, and isolated nodes."""
        graph = nx.Graph()
        graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 0),
                (3, 4),
                (4, 5),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 6),
            ]
        )
        graph.add_node(10)

        result = cdu.compute_LRC(graph, attribute="curvature")

        self.assertIs(result, graph)
        self.assertEqual(graph.number_of_nodes(), 11)
        self.assertEqual(graph.number_of_edges(), 9)
        for u, v in [(0, 1), (1, 2), (2, 0)]:
            self.assertAlmostEqual(graph[u][v]["curvature"], 1.5)
        for u, v in [(3, 4), (4, 5)]:
            self.assertAlmostEqual(graph[u][v]["curvature"], 1.0)
        for u, v in [(6, 7), (7, 8), (8, 9), (9, 6)]:
            self.assertAlmostEqual(graph[u][v]["curvature"], 0.0)

    def test_community_metrics_reuses_neighborhood_cache_across_calls(
        self,
    ) -> None:
        """Check cached neighborhoods preserve metrics and avoid graph lookups."""
        graph = self.make_graph()
        community = set(range(12))
        expected = cdu.compute_community_metrics(graph, community)
        neighborhood_cache = {}

        first = cdu.compute_community_metrics(
            graph,
            community,
            neighborhood_cache=neighborhood_cache,
        )
        self.assertEqual(first, expected)
        self.assertTrue(neighborhood_cache)

        with patch.object(
            graph,
            "neighbors",
            wraps=graph.neighbors,
        ) as neighbor_lookup:
            cached = cdu.compute_community_metrics(
                graph,
                community,
                neighborhood_cache=neighborhood_cache,
            )

        self.assertEqual(cached, expected)
        neighbor_lookup.assert_not_called()

        label_cache = {}
        expected_label = cdu.compute_label_metrics(graph, label=0)
        first_label = cdu.compute_label_metrics(
            graph,
            label=0,
            neighborhood_cache=label_cache,
        )
        self.assertEqual(first_label, expected_label)

        with patch.object(
            graph,
            "neighbors",
            wraps=graph.neighbors,
        ) as neighbor_lookup:
            cached_label = cdu.compute_label_metrics(
                graph,
                label=0,
                neighborhood_cache=label_cache,
            )

        self.assertEqual(cached_label, expected_label)
        neighbor_lookup.assert_not_called()

    def test_sensitivity_reuses_lrc_and_flow_across_orc_percentages(self) -> None:
        """Check shared LRC/flow call counts, output order, and runtime totals."""
        graph = self.make_graph()

        with (
            patch.object(
                cdu,
                "compute_LRC",
                wraps=cdu.compute_LRC,
            ) as compute_lrc,
            patch.object(
                cdu,
                "run_orc_flow_gcc",
                side_effect=self.fake_orc_flow,
            ) as run_flow,
        ):
            results = cdu.run_combo_sensitivity(
                graph=graph,
                lrc_pcts=[0, 10],
                orc_pcts=[0, 50],
                alphas=[0.3, 0.5],
                seeds=[1],
                verbose=False,
            )

        self.assertEqual(compute_lrc.call_count, 1)
        self.assertEqual(run_flow.call_count, 4)
        self.assertEqual(len(results), 8)
        self.assertTrue((results["status"] == "ok").all())
        self.assertEqual(
            list(
                zip(
                    results["alpha"],
                    results["lrc_pct"],
                    results["orc_pct"],
                )
            ),
            [
                (alpha, lrc_pct, orc_pct)
                for alpha in [0.3, 0.5]
                for lrc_pct in [0, 10]
                for orc_pct in [0, 50]
            ],
        )
        self.assertTrue(
            (
                results["runtime_sec"]
                == results["prep_time"] + results["community_time"]
            ).all()
        )

    def test_heatmap_reuses_lrc_and_flow_across_orc_percentages(self) -> None:
        """Check heatmap cells share LRC and one flow per LRC cutoff."""
        graph = self.make_graph()

        with (
            patch.object(
                cdu,
                "compute_LRC",
                wraps=cdu.compute_LRC,
            ) as compute_lrc,
            patch.object(
                cdu,
                "run_orc_flow_gcc",
                side_effect=self.fake_orc_flow,
            ) as run_flow,
        ):
            raw_results, _ = cdu.run_real_dataset_heatmap(
                graph=graph,
                dataset_name="test",
                orc_iterations=1,
                lrc_pcts=[0, 10],
                orc_pcts=[0, 25, 50],
                seeds=[1],
                alpha=0.3,
                plot=False,
            )

        self.assertEqual(compute_lrc.call_count, 1)
        self.assertEqual(run_flow.call_count, 2)
        self.assertEqual(len(raw_results), 6)
        self.assertTrue((raw_results["status"] == "ok").all())
        self.assertTrue(
            (
                raw_results["Runtime"]
                == raw_results["Prep_time"] + raw_results["Community_time"]
            ).all()
        )

    def test_heatmap_preserves_no_flow_zero_cutoffs(self) -> None:
        """Check zero-ORC cells stay unflowed alongside positive cutoffs."""
        graph = self.make_graph()
        evaluated_graphs = {}
        evaluate_prepared_graph = cdu.evaluate_prepared_graph

        def capture_evaluation(info, **kwargs):
            evaluated_graphs[info["method_base"]] = info["graph"].copy()
            return evaluate_prepared_graph(info, **kwargs)

        with (
            patch.object(
                cdu,
                "run_orc_flow_gcc",
                side_effect=self.fake_orc_flow,
            ) as run_flow,
            patch.object(
                cdu,
                "evaluate_prepared_graph",
                side_effect=capture_evaluation,
            ),
        ):
            raw_results, _ = cdu.run_real_dataset_heatmap(
                graph=graph,
                dataset_name="test",
                lrc_pcts=[0, 10],
                orc_pcts=[0, 50],
                seeds=[1],
                plot=False,
            )

        self.assertEqual(run_flow.call_count, 2)
        self.assertEqual(
            set(raw_results["method"]),
            {
                "Plain Louvain",
                "LRC-only",
                "ORC-only",
                "LRC→ORC combo",
            },
        )
        self.assertNotIn(
            "ricciCurvature",
            next(iter(evaluated_graphs["Plain Louvain"].edges(data=True)))[2],
        )
        self.assertNotIn(
            "ricciCurvature",
            next(iter(evaluated_graphs["LRC-only"].edges(data=True)))[2],
        )

    def test_single_configuration_no_flow_path_is_preserved(self) -> None:
        """Check standalone preparation still honors the no-flow option."""
        graph = self.make_graph()

        with (
            patch.object(
                cdu,
                "run_orc_no_flow",
                side_effect=self.fake_orc_flow,
            ) as run_no_flow,
            patch.object(cdu, "run_orc_flow_gcc") as run_flow,
        ):
            info = cdu.prepare_combo_graph(
                graph=graph,
                lrc_pct=0,
                orc_pct=0,
                use_ricci_flow=False,
            )

        run_no_flow.assert_called_once()
        run_flow.assert_not_called()
        self.assertEqual(info["method_base"], "LRC→ORC combo (No Flow)")

    def test_non_gcc_components_are_reattached_after_orc_pruning(self) -> None:
        """Check ORC pruning leaves the non-GCC LRC component untouched."""
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 3), (4, 5)])

        with patch.object(
            cdu,
            "run_orc_flow_gcc",
            side_effect=self.fake_orc_flow,
        ):
            info = cdu.prepare_combo_graph(
                graph=graph,
                lrc_pct=0,
                orc_pct=100,
                iterations=1,
            )

        self.assertEqual(set(info["graph"].edges()), {(4, 5)})


if __name__ == "__main__":
    unittest.main()
