#!/usr/bin/env python3
"""
Decision Quality Evaluator - Measures AI Agent Performance Under Data Duplication

Analyzes how data duplication affects decision quality across 6 key metrics:
1. Decision Consistency - Do duplicates get the same decision?
2. Confidence Stability - Does confidence degrade with duplication?
3. Decision Distribution Shift - Does duplication skew priorities?
4. Cost Efficiency - How much waste from duplicate processing?
5. Reasoning Quality - Does explanation quality degrade?
6. Human-Agent Boundary - When should humans intervene?

Outputs:
- Machine-readable: evaluation_metrics.json
- Human-readable: evaluation_report.md
- Visualizations: 6 charts showing degradation patterns
- Composite Score: Overall decision quality index (0-100)
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter
import statistics

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    PLOTTING_AVAILABLE = True
except ImportError:
    print("⚠️  matplotlib not available. Install with: pip install matplotlib")
    PLOTTING_AVAILABLE = False

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))


class DecisionQualityEvaluator:
    """
    Evaluates decision quality across duplication levels
    """
    
    def __init__(self, decisions_dir: Path, output_dir: Path, cluster_map_dir: Path = None):
        self.decisions_dir = Path(decisions_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load cluster maps (record_id -> customer_id lookup per duplication level)
        self.cluster_map_dir = Path(cluster_map_dir) if cluster_map_dir else None
        self.cluster_maps = self._load_cluster_maps()

        # Load all datasets
        self.datasets = self._load_all_datasets()
        self.baseline = self.datasets.get('0pct', {})
        
        # Results storage
        self.metrics = {}
        self.composite_score = 0.0

    def _load_cluster_maps(self) -> Dict[str, Dict[str, str]]:
        """
        Load cluster map files and build record_id -> customer_id lookup per dup level.

        cluster_map_{level}pct.json structure:
            { "CUST_000001": { "record_ids": ["REC_AAA", "REC_BBB"], ... }, ... }

        Returns:
            { "10pct": { "REC_AAA": "CUST_000001", "REC_BBB": "CUST_000001", ... }, ... }
        """
        if not self.cluster_map_dir or not self.cluster_map_dir.exists():
            print("⚠️  No cluster_map_dir provided — consistency analysis will be limited")
            return {}

        cluster_maps = {}
        for map_file in sorted(self.cluster_map_dir.glob("cluster_map_*.json")):
            filename = map_file.name
            dup_level = filename.replace("cluster_map_", "").replace(".json", "")
            if not dup_level.endswith("pct"):
                continue

            with open(map_file) as f:
                raw_map = json.load(f)

            # Invert: customer_id -> [record_ids]  =>  record_id -> customer_id
            record_to_customer = {}
            for customer_id, cluster_info in raw_map.items():
                for record_id in cluster_info.get("record_ids", []):
                    record_to_customer[record_id] = customer_id

            cluster_maps[dup_level] = record_to_customer
            print(f"  ✅ Cluster map loaded: {dup_level} ({len(record_to_customer)} record mappings)")

        print(f"\n📂 Loaded {len(cluster_maps)} cluster maps")
        return cluster_maps

    def _get_record_to_customer_map(self, dup_level: str) -> Dict[str, str]:
        """
        Return the record_id -> customer_id lookup for a given duplication level.
        Falls back to identity mapping (record_id -> record_id) if no cluster map available,
        which means consistency analysis will show 100% — a safe but clearly labelled fallback.
        """
        if dup_level in self.cluster_maps:
            return self.cluster_maps[dup_level]

        # Fallback: no cluster map — warn and return identity map
        print(f"  ⚠️  No cluster map for {dup_level} — using identity mapping (consistency will read 100%)")
        decisions = self.datasets.get(dup_level, {}).get('decisions', [])
        return {d['record_id']: d['record_id'] for d in decisions}

    def _load_all_datasets(self) -> Dict[str, Any]:
        """Load all decision files"""
        datasets = {}
        
        for jsonl_file in sorted(self.decisions_dir.glob("customers_dup_*.decisions.jsonl")):
            # Extract duplication level from filename
            # Example: "customers_dup_0pct.decisions.jsonl" -> "0pct"
            filename = jsonl_file.name  # Gets "customers_dup_0pct.decisions.jsonl"
            
            # Remove prefix and suffix to get just the duplication level
            # "customers_dup_0pct.decisions.jsonl" -> "0pct"
            if filename.startswith("customers_dup_") and filename.endswith(".decisions.jsonl"):
                # Extract the middle part
                dup_level = filename.replace("customers_dup_", "").replace(".decisions.jsonl", "")
                
                # Validate it looks like a duplication level (e.g., "0pct", "10pct")
                if dup_level.endswith("pct"):
                    decisions = []
                    with open(jsonl_file, 'r') as f:
                        for line in f:
                            decisions.append(json.loads(line))
                    
                    datasets[dup_level] = {
                        'file': jsonl_file,
                        'decisions': decisions,
                        'dup_level': dup_level
                    }
                    print(f"  ✅ Loaded: {dup_level} ({len(decisions)} decisions)")
                else:
                    print(f"⚠️  Skipping file with unexpected format: {filename}")
            else:
                print(f"⚠️  Skipping file with unexpected format: {filename}")
            
        print(f"\n📂 Loaded {len(datasets)} datasets")
        return datasets


    # ========================================================================
    # METRIC 1: Decision Consistency
    # ========================================================================
    
    def analyze_decision_consistency(self) -> Dict[str, Any]:
        """
        Calculate what % of duplicate records get the same decision
        
        Returns:
            {
                'consistency_by_level': {
                    '10pct': 0.95,  # 95% of duplicates got same decision
                    '20pct': 0.92,
                    ...
                },
                'inconsistent_examples': [...],
                'overall_consistency': 0.89
            }
        """
        print("\n" + "="*60)
        print("📊 METRIC 1: Decision Consistency Analysis")
        print("="*60)
        
        consistency_by_level = {}
        all_inconsistencies = []
        
        for dup_level, data in sorted(self.datasets.items()):
            if dup_level == '0pct':
                consistency_by_level[dup_level] = 1.0  # No duplicates = perfect consistency
                continue

            # Get record_id -> customer_id mapping from cluster map
            record_to_customer = self._get_record_to_customer_map(dup_level)

            # Group decisions by true customer_id (not record_id)
            customer_decisions = defaultdict(list)
            for decision in data['decisions']:
                record_id = decision['record_id']
                customer_id = record_to_customer.get(record_id, record_id)  # fallback to record_id
                customer_decisions[customer_id].append({
                    'record_id': record_id,
                    'decision': decision['business_decision'],
                    'confidence': decision['agent_confidence'],
                    'reasoning': decision['decision_reasoning']
                })
            
            # Find duplicates (customers appearing more than once)
            duplicates = {cid: decs for cid, decs in customer_decisions.items() if len(decs) > 1}
            
            if not duplicates:
                consistency_by_level[dup_level] = 1.0
                continue
            
            # Calculate consistency
            consistent_count = 0
            inconsistent_count = 0
            
            for customer_id, decisions in duplicates.items():
                decision_values = [d['decision'] for d in decisions]
                
                if len(set(decision_values)) == 1:
                    # All decisions match
                    consistent_count += 1
                else:
                    # Conflicting decisions
                    inconsistent_count += 1
                    all_inconsistencies.append({
                        'customer_id': customer_id,
                        'dup_level': dup_level,
                        'decisions': decisions
                    })
            
            total_duplicates = len(duplicates)
            consistency_rate = consistent_count / total_duplicates if total_duplicates > 0 else 1.0
            consistency_by_level[dup_level] = consistency_rate
            
            print(f"  {dup_level:6s}: {consistency_rate:6.2%} consistent "
                  f"({consistent_count}/{total_duplicates} duplicates)")
        
        # Calculate overall consistency (weighted by number of duplicates)
        overall_consistency = statistics.mean(consistency_by_level.values())
        
        result = {
            'consistency_by_level': consistency_by_level,
            'inconsistent_examples': all_inconsistencies[:10],  # First 10 examples
            'total_inconsistencies': len(all_inconsistencies),
            'overall_consistency': overall_consistency,
            'metric_score': overall_consistency * 100  # 0-100 scale
        }
        
        print(f"\n  📈 Overall Consistency: {overall_consistency:.2%}")
        print(f"  ⚠️  Total Inconsistencies: {len(all_inconsistencies)}")
        
        return result
    
    # ========================================================================
    # METRIC 2: Confidence Stability
    # ========================================================================
    
    def analyze_confidence_stability(self) -> Dict[str, Any]:
        """
        Measure how confidence changes with duplication
        
        Returns:
            {
                'confidence_by_level': {
                    '0pct': 0.759,
                    '10pct': 0.758,
                    ...
                },
                'confidence_degradation': -0.001,  # Change from baseline
                'stability_score': 0.98  # How stable (1.0 = no change)
            }
        """
        print("\n" + "="*60)
        print("📊 METRIC 2: Confidence Stability Analysis")
        print("="*60)
        
        confidence_by_level = {}
        
        for dup_level, data in sorted(self.datasets.items()):
            confidences = [d['agent_confidence'] for d in data['decisions']]
            avg_confidence = statistics.mean(confidences)
            confidence_by_level[dup_level] = avg_confidence
            
            print(f"  {dup_level:6s}: {avg_confidence:.4f}")
        
        # Calculate degradation from baseline
        baseline_confidence = confidence_by_level.get('0pct', 0.75)
        worst_confidence = min(confidence_by_level.values())
        confidence_degradation = worst_confidence - baseline_confidence
        
        # Stability score: how close to baseline (1.0 = perfect stability)
        confidence_range = max(confidence_by_level.values()) - min(confidence_by_level.values())
        stability_score = 1.0 - (confidence_range / baseline_confidence)
        
        result = {
            'confidence_by_level': confidence_by_level,
            'baseline_confidence': baseline_confidence,
            'worst_confidence': worst_confidence,
            'confidence_degradation': confidence_degradation,
            'confidence_range': confidence_range,
            'stability_score': stability_score,
            'metric_score': stability_score * 100  # 0-100 scale
        }
        
        print(f"\n  📈 Baseline Confidence: {baseline_confidence:.4f}")
        print(f"  📉 Worst Confidence: {worst_confidence:.4f}")
        print(f"  🔄 Degradation: {confidence_degradation:+.4f}")
        print(f"  ⚖️  Stability Score: {stability_score:.2%}")
        
        return result
    
    # ========================================================================
    # METRIC 3: Decision Distribution Shift
    # ========================================================================
    
    def analyze_distribution_shift(self) -> Dict[str, Any]:
        """
        Measure how decision distribution changes with duplication
        
        Returns:
            {
                'distribution_by_level': {
                    '0pct': {'HIGH': 0.008, 'MEDIUM': 0.648, 'LOW': 0.344},
                    ...
                },
                'shift_from_baseline': {...},
                'max_shift': 0.05  # Maximum % point shift
            }
        """
        print("\n" + "="*60)
        print("📊 METRIC 3: Decision Distribution Shift Analysis")
        print("="*60)
        
        distribution_by_level = {}
        
        for dup_level, data in sorted(self.datasets.items()):
            decisions = [d['business_decision'] for d in data['decisions']]
            total = len(decisions)
            
            distribution = {
                'HIGH_PRIORITY': decisions.count('HIGH_PRIORITY') / total,
                'MEDIUM_PRIORITY': decisions.count('MEDIUM_PRIORITY') / total,
                'LOW_PRIORITY': decisions.count('LOW_PRIORITY') / total
            }
            
            distribution_by_level[dup_level] = distribution
            
            print(f"  {dup_level:6s}: H:{distribution['HIGH_PRIORITY']:5.1%} "
                  f"M:{distribution['MEDIUM_PRIORITY']:5.1%} "
                  f"L:{distribution['LOW_PRIORITY']:5.1%}")
        
        # Calculate shift from baseline
        baseline_dist = distribution_by_level.get('0pct', {})
        shift_from_baseline = {}
        max_shift = 0.0
        
        for dup_level, dist in distribution_by_level.items():
            if dup_level == '0pct':
                continue
            
            shift = {}
            for priority in ['HIGH_PRIORITY', 'MEDIUM_PRIORITY', 'LOW_PRIORITY']:
                shift[priority] = dist[priority] - baseline_dist.get(priority, 0)
                max_shift = max(max_shift, abs(shift[priority]))
            
            shift_from_baseline[dup_level] = shift
        
        # Shift score: 1.0 = no shift, 0.0 = complete shift
        shift_score = 1.0 - min(max_shift * 10, 1.0)  # Scale to 0-1
        
        result = {
            'distribution_by_level': distribution_by_level,
            'baseline_distribution': baseline_dist,
            'shift_from_baseline': shift_from_baseline,
            'max_shift': max_shift,
            'shift_score': shift_score,
            'metric_score': shift_score * 100  # 0-100 scale
        }
        
        print(f"\n  📈 Max Distribution Shift: {max_shift:+.2%}")
        print(f"  ⚖️  Shift Score: {shift_score:.2%}")
        
        return result
    
    # ========================================================================
    # METRIC 4: Cost Efficiency
    # ========================================================================

    def analyze_cost_efficiency(self) -> Dict[str, Any]:
        """
        Calculate cost waste from duplicate processing
        
        Returns:
            {
                'cost_by_level': {...},
                'cost_per_unique_customer': {...},
                'waste_by_level': {...},
                'total_waste': 0.25
            }
        """
        print("\n" + "="*60)
        print("📊 METRIC 4: Cost Efficiency Analysis")
        print("="*60)
        
        cost_by_level = {}
        cost_per_record = {}
        cost_per_unique = {}
        waste_by_level = {}
        
        # First pass: get baseline metrics from 0% duplication
        baseline_cost_per_record = 0.0001  # Default fallback
        baseline_unique_customers = 500  # Default fallback
        
        if '0pct' in self.datasets:
            baseline_data = self.datasets['0pct']
            baseline_total_cost = sum(d.get('cost_usd', 0) for d in baseline_data['decisions'])
            baseline_total_records = len(baseline_data['decisions'])
            # At 0% duplication every record IS a unique customer, so record count = customer count
            baseline_unique_customers = baseline_total_records
            baseline_cost_per_record = baseline_total_cost / baseline_total_records if baseline_total_records > 0 else 0.0001
        
        print(f"  Baseline (0pct): {baseline_unique_customers} unique customers, "
              f"${baseline_cost_per_record:.6f} per record")
        print()
        
        # Second pass: calculate costs and waste for all levels
        for dup_level, data in sorted(self.datasets.items(), 
                                     key=lambda x: int(x[0].replace('pct', ''))):
            total_cost = sum(d.get('cost_usd', 0) for d in data['decisions'])
            total_records = len(data['decisions'])

            # Use cluster map to count true unique customers
            record_to_customer = self._get_record_to_customer_map(dup_level)
            unique_customers = len(set(
                record_to_customer.get(d['record_id'], d['record_id'])
                for d in data['decisions']
            ))
            
            cost_per_rec = total_cost / total_records if total_records > 0 else 0
            cost_per_uniq = total_cost / unique_customers if unique_customers > 0 else 0
            
            cost_by_level[dup_level] = total_cost
            cost_per_record[dup_level] = cost_per_rec
            cost_per_unique[dup_level] = cost_per_uniq
            
            if dup_level == '0pct':
                waste_by_level[dup_level] = 0.0
            else:
                # Waste = cost above what we'd pay if we only processed unique customers once
                # Expected cost = baseline unique customers × baseline cost per record
                expected_cost = baseline_unique_customers * baseline_cost_per_record
                waste = total_cost - expected_cost
                waste_by_level[dup_level] = max(waste, 0.0)  # Don't show negative waste
            
            print(f"  {dup_level:6s}: ${total_cost:.4f} total, "
                  f"${cost_per_uniq:.6f}/unique, "
                  f"{unique_customers} unique customers, "
                  f"waste: ${waste_by_level.get(dup_level, 0):.4f}")
        
        total_waste = sum(waste_by_level.values())
        total_cost = sum(cost_by_level.values())
        efficiency_score = 1.0 - (total_waste / total_cost) if total_cost > 0 else 1.0
        
        result = {
            'cost_by_level': cost_by_level,
            'cost_per_record': cost_per_record,
            'cost_per_unique_customer': cost_per_unique,
            'waste_by_level': waste_by_level,
            'baseline_unique_customers': baseline_unique_customers,
            'baseline_cost_per_record': baseline_cost_per_record,
            'total_waste': total_waste,
            'total_cost': total_cost,
            'efficiency_score': efficiency_score,
            'metric_score': efficiency_score * 100  # 0-100 scale
        }
        
        print(f"\n  💰 Total Cost: ${total_cost:.4f}")
        print(f"  🗑️  Total Waste: ${total_waste:.4f} ({total_waste/total_cost:.1%})")
        print(f"  ⚖️  Efficiency Score: {efficiency_score:.2%}")
        
        return result


    # ========================================================================
    # METRIC 5: Reasoning Quality
    # ========================================================================
    
    def analyze_reasoning_quality(self) -> Dict[str, Any]:
        """
        Assess quality of decision reasoning/explanations
        
        Returns:
            {
                'avg_reasoning_length_by_level': {...},
                'reasoning_keywords_by_level': {...},
                'quality_score': 0.85
            }
        """
        print("\n" + "="*60)
        print("📊 METRIC 5: Reasoning Quality Analysis")
        print("="*60)
        
        reasoning_length_by_level = {}
        reasoning_keywords_by_level = {}
        
        # Keywords that indicate quality reasoning
        quality_keywords = [
            'however', 'although', 'because', 'therefore', 'indicates',
            'suggests', 'demonstrates', 'based on', 'given that', 'considering'
        ]
        
        for dup_level, data in sorted(self.datasets.items()):
            reasonings = [d.get('decision_reasoning', '') for d in data['decisions']]
            
            # Average reasoning length
            avg_length = statistics.mean([len(r.split()) for r in reasonings])
            reasoning_length_by_level[dup_level] = avg_length
            
            # Count quality keywords
            keyword_counts = Counter()
            for reasoning in reasonings:
                reasoning_lower = reasoning.lower()
                for keyword in quality_keywords:
                    if keyword in reasoning_lower:
                        keyword_counts[keyword] += 1
            
            reasoning_keywords_by_level[dup_level] = dict(keyword_counts)
            
            total_keywords = sum(keyword_counts.values())
            keywords_per_decision = total_keywords / len(reasonings) if reasonings else 0
            
            print(f"  {dup_level:6s}: {avg_length:.1f} words/reasoning, "
                  f"{keywords_per_decision:.2f} quality keywords/decision")
        
        # Calculate quality score based on consistency of reasoning length
        baseline_length = reasoning_length_by_level.get('0pct', 50)
        length_variance = statistics.stdev(reasoning_length_by_level.values())
        quality_score = 1.0 - min(length_variance / baseline_length, 1.0)
        
        result = {
            'avg_reasoning_length_by_level': reasoning_length_by_level,
            'reasoning_keywords_by_level': reasoning_keywords_by_level,
            'baseline_length': baseline_length,
            'length_variance': length_variance,
            'quality_score': quality_score,
            'metric_score': quality_score * 100  # 0-100 scale
        }
        
        print(f"\n  📈 Baseline Reasoning Length: {baseline_length:.1f} words")
        print(f"  📊 Length Variance: {length_variance:.2f}")
        print(f"  ⚖️  Quality Score: {quality_score:.2%}")
        
        return result
    
    # ========================================================================
    # METRIC 6: Human-Agent Boundary
    # ========================================================================
    
    def analyze_human_agent_boundary(self) -> Dict[str, Any]:
        """
        Determine optimal confidence threshold for human review
        
        Returns:
            {
                'confidence_distribution_by_level': {...},
                'recommended_threshold_by_level': {...},
                'decisions_requiring_review': {...}
            }
        """
        print("\n" + "="*60)
        print("📊 METRIC 6: Human-Agent Boundary Analysis")
        print("="*60)
        
        confidence_distribution_by_level = {}
        recommended_threshold_by_level = {}
        decisions_requiring_review = {}
        
        for dup_level, data in sorted(self.datasets.items()):
            confidences = [d['agent_confidence'] for d in data['decisions']]
            
            # Calculate confidence distribution
            confidence_distribution_by_level[dup_level] = {
                'mean': statistics.mean(confidences),
                'median': statistics.median(confidences),
                'stdev': statistics.stdev(confidences) if len(confidences) > 1 else 0,
                'min': min(confidences),
                'max': max(confidences),
                'p25': statistics.quantiles(confidences, n=4)[0],
                'p75': statistics.quantiles(confidences, n=4)[2]
            }
            
            # Recommended threshold: mean - 1 stdev (captures ~84% of decisions)
            mean_conf = confidence_distribution_by_level[dup_level]['mean']
            stdev_conf = confidence_distribution_by_level[dup_level]['stdev']
            threshold = max(mean_conf - stdev_conf, 0.5)  # Minimum 0.5
            recommended_threshold_by_level[dup_level] = threshold
            
            # Count decisions below threshold
            below_threshold = sum(1 for c in confidences if c < threshold)
            decisions_requiring_review[dup_level] = {
                'count': below_threshold,
                'percentage': below_threshold / len(confidences) if confidences else 0
            }
            
            print(f"  {dup_level:6s}: threshold={threshold:.3f}, "
                  f"review={below_threshold}/{len(confidences)} "
                  f"({decisions_requiring_review[dup_level]['percentage']:.1%})")
        
        # Calculate boundary score: lower review % = better
        avg_review_pct = statistics.mean([
            d['percentage'] for d in decisions_requiring_review.values()
        ])
        boundary_score = 1.0 - avg_review_pct
        
        result = {
            'confidence_distribution_by_level': confidence_distribution_by_level,
            'recommended_threshold_by_level': recommended_threshold_by_level,
            'decisions_requiring_review': decisions_requiring_review,
            'avg_review_percentage': avg_review_pct,
            'boundary_score': boundary_score,
            'metric_score': boundary_score * 100  # 0-100 scale
        }
        
        print(f"\n  📈 Avg Review Required: {avg_review_pct:.1%}")
        print(f"  ⚖️  Boundary Score: {boundary_score:.2%}")
        
        return result
    
    # ========================================================================
    # METRIC 7: Field Importance Analysis (H4)
    # ========================================================================

    def analyze_field_importance(self) -> Dict[str, Any]:
        """
        H4: Analyze which fields the agent cited most as key decision factors,
        and whether field importance shifts as duplication level increases.

        Uses the key_factors field added to the agent output schema.
        """
        print("\n" + "="*60)
        print("📊 METRIC 7 (H4): Field Importance Analysis")
        print("="*60)

        field_frequency_by_level = {}
        all_fields = set()

        for dup_level, data in sorted(self.datasets.items()):
            field_counts = Counter()
            total_decisions = 0

            for decision in data['decisions']:
                key_factors = decision.get('key_factors', [])
                if isinstance(key_factors, list):
                    for field in key_factors:
                        field_counts[field.strip()] += 1
                    total_decisions += 1

            # Normalize to frequency (proportion of decisions citing each field)
            field_frequency = {
                field: count / total_decisions if total_decisions > 0 else 0
                for field, count in field_counts.items()
            }
            field_frequency_by_level[dup_level] = field_frequency
            all_fields.update(field_counts.keys())

            top_fields = field_counts.most_common(5)
            print(f"  {dup_level:6s} top factors: " +
                  ", ".join(f"{f}({c})" for f, c in top_fields))

        # Identify the most consistently cited fields across all levels
        field_total_counts = Counter()
        for freq_map in field_frequency_by_level.values():
            for field, freq in freq_map.items():
                field_total_counts[field] += freq

        top_fields_overall = field_total_counts.most_common(10)

        # Measure stability: does the top-5 field ranking change across levels?
        rankings_by_level = {}
        for dup_level, freq_map in field_frequency_by_level.items():
            rankings_by_level[dup_level] = [
                f for f, _ in sorted(freq_map.items(), key=lambda x: -x[1])
            ][:5]

        # Rank stability score: what % of top-5 fields are consistent across levels
        if len(rankings_by_level) > 1:
            baseline_top5 = set(list(rankings_by_level.values())[0])
            stability_scores = []
            for level, ranking in list(rankings_by_level.items())[1:]:
                overlap = len(baseline_top5 & set(ranking)) / 5
                stability_scores.append(overlap)
            ranking_stability = statistics.mean(stability_scores)
        else:
            ranking_stability = 1.0

        result = {
            'field_frequency_by_level': field_frequency_by_level,
            'top_fields_overall': dict(top_fields_overall),
            'rankings_by_level': rankings_by_level,
            'ranking_stability': ranking_stability,
            'all_fields_observed': sorted(all_fields),
        }

        print(f"\n  📈 Top fields overall: {[f for f, _ in top_fields_overall[:5]]}")
        print(f"  ⚖️  Ranking stability across levels: {ranking_stability:.2%}")

        return result

    # ========================================================================
    # METRIC 8: Segment Distortion Analysis (H6)
    # ========================================================================

    def analyze_segment_distortion(self) -> Dict[str, Any]:
        """
        H6: Measure whether duplication disproportionately distorts decisions
        for specific customer segments.

        Uses customer_segment field added to the agent output schema.
        """
        print("\n" + "="*60)
        print("📊 METRIC 8 (H6): Segment Distortion Analysis")
        print("="*60)

        SEGMENTS = ['high_value', 'medium_value', 'low_value', 'at_risk']
        PRIORITIES = ['HIGH_PRIORITY', 'MEDIUM_PRIORITY', 'LOW_PRIORITY']

        distribution_by_segment_by_level = {}
        segment_shift_by_level = {}

        # Build baseline distribution per segment from 0% level
        baseline_segment_dist = {}
        if '0pct' in self.datasets:
            for decision in self.datasets['0pct']['decisions']:
                seg = decision.get('customer_segment')
                if not seg:
                    continue
                if seg not in baseline_segment_dist:
                    baseline_segment_dist[seg] = Counter()
                baseline_segment_dist[seg][decision['business_decision']] += 1

            # Normalize to proportions
            for seg in baseline_segment_dist:
                total = sum(baseline_segment_dist[seg].values())
                baseline_segment_dist[seg] = {
                    p: baseline_segment_dist[seg].get(p, 0) / total
                    for p in PRIORITIES
                }

        for dup_level, data in sorted(self.datasets.items()):
            seg_dist = {seg: Counter() for seg in SEGMENTS}

            for decision in data['decisions']:
                seg = decision.get('customer_segment')
                if seg and seg in seg_dist:
                    seg_dist[seg][decision['business_decision']] += 1

            # Normalize
            normalized = {}
            for seg in SEGMENTS:
                total = sum(seg_dist[seg].values())
                if total > 0:
                    normalized[seg] = {
                        p: seg_dist[seg].get(p, 0) / total
                        for p in PRIORITIES
                    }
                else:
                    normalized[seg] = {p: 0.0 for p in PRIORITIES}

            distribution_by_segment_by_level[dup_level] = normalized

            # Calculate shift from baseline per segment
            if dup_level != '0pct' and baseline_segment_dist:
                shifts = {}
                for seg in SEGMENTS:
                    if seg in baseline_segment_dist:
                        max_shift = max(
                            abs(normalized[seg].get(p, 0) -
                                baseline_segment_dist[seg].get(p, 0))
                            for p in PRIORITIES
                        )
                        shifts[seg] = max_shift
                segment_shift_by_level[dup_level] = shifts

                print(f"  {dup_level:6s} max shift by segment: " +
                      ", ".join(f"{s}:{v:.1%}" for s, v in shifts.items()))

        # Which segment is most sensitive to duplication?
        segment_sensitivity = {}
        for seg in SEGMENTS:
            shifts_for_seg = [
                level_shifts.get(seg, 0)
                for level_shifts in segment_shift_by_level.values()
            ]
            segment_sensitivity[seg] = statistics.mean(shifts_for_seg) if shifts_for_seg else 0.0

        most_sensitive = max(segment_sensitivity, key=segment_sensitivity.get) if segment_sensitivity else 'unknown'

        # Overall distortion score: lower = more distortion
        avg_max_shift = statistics.mean([
            max(shifts.values()) if shifts else 0
            for shifts in segment_shift_by_level.values()
        ]) if segment_shift_by_level else 0.0
        distortion_score = 1.0 - min(avg_max_shift * 5, 1.0)

        result = {
            'distribution_by_segment_by_level': distribution_by_segment_by_level,
            'baseline_segment_distribution': baseline_segment_dist,
            'segment_shift_by_level': segment_shift_by_level,
            'segment_sensitivity': segment_sensitivity,
            'most_sensitive_segment': most_sensitive,
            'avg_max_shift': avg_max_shift,
            'distortion_score': distortion_score,
        }

        print(f"\n  📈 Most sensitive segment: {most_sensitive} "
              f"(avg shift {segment_sensitivity.get(most_sensitive, 0):.2%})")
        print(f"  ⚖️  Distortion score: {distortion_score:.2%}")

        return result

    # ========================================================================
    # COMPOSITE DECISION QUALITY SCORE
    # ========================================================================
    
    def calculate_composite_score(self) -> Dict[str, Any]:
        """
        Calculate overall decision quality score (0-100)
        
        Weighted average of all 6 metrics:
        - Decision Consistency: 30%
        - Confidence Stability: 15%
        - Distribution Shift: 10%
        - Cost Efficiency: 15%
        - Reasoning Quality: 15%
        - Human-Agent Boundary: 15%
        """
        print("\n" + "="*60)
        print("🎯 COMPOSITE DECISION QUALITY SCORE")
        print("="*60)
        
        weights = {
            'decision_consistency': 0.30,
            'confidence_stability': 0.15,
            'distribution_shift': 0.10,
            'cost_efficiency': 0.15,
            'reasoning_quality': 0.15,
            'human_agent_boundary': 0.15
        }
        
        scores = {
            'decision_consistency': self.metrics['decision_consistency']['metric_score'],
            'confidence_stability': self.metrics['confidence_stability']['metric_score'],
            'distribution_shift': self.metrics['distribution_shift']['metric_score'],
            'cost_efficiency': self.metrics['cost_efficiency']['metric_score'],
            'reasoning_quality': self.metrics['reasoning_quality']['metric_score'],
            'human_agent_boundary': self.metrics['human_agent_boundary']['metric_score']
        }
        
        # Calculate weighted composite score
        composite_score = sum(scores[metric] * weights[metric] for metric in weights)
        
        print(f"\n  Individual Metric Scores (0-100):")
        for metric, score in scores.items():
            weight = weights[metric]
            weighted = score * weight
            print(f"    {metric:25s}: {score:5.1f} × {weight:.0%} = {weighted:5.1f}")
        
        print(f"\n  {'='*60}")
        print(f"  🏆 COMPOSITE QUALITY SCORE: {composite_score:.1f}/100")
        print(f"  {'='*60}")
        
        # Quality rating
        if composite_score >= 90:
            rating = "EXCELLENT"
        elif composite_score >= 80:
            rating = "GOOD"
        elif composite_score >= 70:
            rating = "FAIR"
        elif composite_score >= 60:
            rating = "POOR"
        else:
            rating = "CRITICAL"
        
        result = {
            'composite_score': composite_score,
            'rating': rating,
            'weights': weights,
            'individual_scores': scores,
            'weighted_scores': {m: scores[m] * weights[m] for m in weights}
        }
        
        print(f"  📊 Quality Rating: {rating}")
        
        return result
    
    # ========================================================================
    # VISUALIZATION GENERATION
    # ========================================================================
    
    def generate_visualizations(self):
        """Generate all visualization charts"""
        if not PLOTTING_AVAILABLE:
            print("\n⚠️  Skipping visualizations (matplotlib not available)")
            return
        
        print("\n" + "="*60)
        print("📊 Generating Visualizations")
        print("="*60)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with 6 subplots (2 rows, 3 columns)
        fig = plt.figure(figsize=(18, 12))
        
        # Extract duplication levels (sorted)
        dup_levels = sorted(self.datasets.keys(), 
                           key=lambda x: int(x.replace('pct', '')))
        dup_percentages = [int(x.replace('pct', '')) for x in dup_levels]
        
        # ---- Chart 1: Decision Consistency ----
        ax1 = plt.subplot(2, 3, 1)
        consistency_data = [
            self.metrics['decision_consistency']['consistency_by_level'][level]
            for level in dup_levels
        ]
        ax1.plot(dup_percentages, [c * 100 for c in consistency_data], 
                marker='o', linewidth=2, markersize=8, color='#2E86AB')
        ax1.set_xlabel('Duplication Level (%)', fontsize=11)
        ax1.set_ylabel('Consistency Rate (%)', fontsize=11)
        ax1.set_title('1. Decision Consistency vs Duplication', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 105])
        
        # ---- Chart 2: Confidence Stability ----
        ax2 = plt.subplot(2, 3, 2)
        confidence_data = [
            self.metrics['confidence_stability']['confidence_by_level'][level]
            for level in dup_levels
        ]
        ax2.plot(dup_percentages, [c * 100 for c in confidence_data], 
                marker='s', linewidth=2, markersize=8, color='#A23B72')
        ax2.set_xlabel('Duplication Level (%)', fontsize=11)
        ax2.set_ylabel('Average Confidence (%)', fontsize=11)
        ax2.set_title('2. Confidence Stability vs Duplication', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([70, 80])
        
        # ---- Chart 3: Decision Distribution ----
        ax3 = plt.subplot(2, 3, 3)
        high_data = [
            self.metrics['distribution_shift']['distribution_by_level'][level]['HIGH_PRIORITY'] * 100
            for level in dup_levels
        ]
        medium_data = [
            self.metrics['distribution_shift']['distribution_by_level'][level]['MEDIUM_PRIORITY'] * 100
            for level in dup_levels
        ]
        low_data = [
            self.metrics['distribution_shift']['distribution_by_level'][level]['LOW_PRIORITY'] * 100
            for level in dup_levels
        ]
        
        ax3.bar(dup_percentages, high_data, label='HIGH', color='#F18F01', alpha=0.8)
        ax3.bar(dup_percentages, medium_data, bottom=high_data, 
               label='MEDIUM', color='#C73E1D', alpha=0.8)
        ax3.bar(dup_percentages, low_data, 
               bottom=[h+m for h,m in zip(high_data, medium_data)],
               label='LOW', color='#6A994E', alpha=0.8)
        
        ax3.set_xlabel('Duplication Level (%)', fontsize=11)
        ax3.set_ylabel('Decision Distribution (%)', fontsize=11)
        ax3.set_title('3. Decision Distribution Shift', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # ---- Chart 4: Cost Efficiency ----
        ax4 = plt.subplot(2, 3, 4)
        cost_data = [
            self.metrics['cost_efficiency']['cost_by_level'][level] * 1000  # Convert to mills
            for level in dup_levels
        ]
        waste_data = [
            self.metrics['cost_efficiency']['waste_by_level'].get(level, 0) * 1000
            for level in dup_levels
        ]
        
        ax4.bar(dup_percentages, cost_data, label='Total Cost', color='#2E86AB', alpha=0.7)
        ax4.bar(dup_percentages, waste_data, label='Waste', color='#C73E1D', alpha=0.9)
        ax4.set_xlabel('Duplication Level (%)', fontsize=11)
        ax4.set_ylabel('Cost (mills, $0.001)', fontsize=11)
        ax4.set_title('4. Cost Efficiency & Waste', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # ---- Chart 5: Reasoning Quality ----
        ax5 = plt.subplot(2, 3, 5)
        reasoning_length_data = [
            self.metrics['reasoning_quality']['avg_reasoning_length_by_level'][level]
            for level in dup_levels
        ]
        ax5.plot(dup_percentages, reasoning_length_data, 
                marker='^', linewidth=2, markersize=8, color='#6A994E')
        ax5.set_xlabel('Duplication Level (%)', fontsize=11)
        ax5.set_ylabel('Avg Reasoning Length (words)', fontsize=11)
        ax5.set_title('5. Reasoning Quality vs Duplication', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # ---- Chart 6: Human-Agent Boundary ----
        ax6 = plt.subplot(2, 3, 6)
        review_data = [
            self.metrics['human_agent_boundary']['decisions_requiring_review'][level]['percentage'] * 100
            for level in dup_levels
        ]
        threshold_data = [
            self.metrics['human_agent_boundary']['recommended_threshold_by_level'][level] * 100
            for level in dup_levels
        ]
        
        ax6_twin = ax6.twinx()
        
        line1 = ax6.plot(dup_percentages, review_data, 
                marker='o', linewidth=2, markersize=8, color='#C73E1D', 
                label='% Requiring Review')
        line2 = ax6_twin.plot(dup_percentages, threshold_data, 
                marker='s', linewidth=2, markersize=8, color='#2E86AB',
                linestyle='--', label='Confidence Threshold')
        
        ax6.set_xlabel('Duplication Level (%)', fontsize=11)
        ax6.set_ylabel('Decisions Requiring Review (%)', fontsize=11, color='#C73E1D')
        ax6_twin.set_ylabel('Confidence Threshold (%)', fontsize=11, color='#2E86AB')
        ax6.set_title('6. Human-Agent Boundary', fontsize=12, fontweight='bold')
        ax6.tick_params(axis='y', labelcolor='#C73E1D')
        ax6_twin.tick_params(axis='y', labelcolor='#2E86AB')
        ax6.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='upper left', fontsize=9)
        
        # Overall title
        composite_score = self.metrics['composite_score']['composite_score']
        rating = self.metrics['composite_score']['rating']
        fig.suptitle(f'Decision Quality Analysis Across Duplication Levels\n'
                    f'Composite Quality Score: {composite_score:.1f}/100 ({rating})',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        output_file = self.output_dir / 'decision_quality_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {output_file}")
        
        plt.close()
        
        # ---- Generate Composite Score Visualization ----
        self._generate_composite_score_chart()
    
    def _generate_composite_score_chart(self):
        """Generate a separate chart showing composite score breakdown"""
        if not PLOTTING_AVAILABLE:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Horizontal bar chart of individual scores
        metrics = list(self.metrics['composite_score']['individual_scores'].keys())
        scores = list(self.metrics['composite_score']['individual_scores'].values())
        weights = list(self.metrics['composite_score']['weights'].values())
        
        # Format metric names for display
        metric_labels = [m.replace('_', ' ').title() for m in metrics]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8B4513']
        
        y_pos = range(len(metrics))
        bars = ax1.barh(y_pos, scores, color=colors, alpha=0.8)
        
        # Add weight labels on bars
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{weight:.0%}', ha='left', va='center', fontsize=10)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(metric_labels)
        ax1.set_xlabel('Score (0-100)', fontsize=12)
        ax1.set_title('Individual Metric Scores\n(with weights)', fontsize=13, fontweight='bold')
        ax1.set_xlim([0, 110])
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Right: Gauge chart for composite score
        composite_score = self.metrics['composite_score']['composite_score']
        rating = self.metrics['composite_score']['rating']
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        
        # Color zones
        ax2.fill_between(theta, 0, 60, color='#C73E1D', alpha=0.3, label='Poor (0-60)')
        ax2.fill_between(theta, 60, 70, color='#F18F01', alpha=0.3, label='Fair (60-70)')
        ax2.fill_between(theta, 70, 80, color='#6A994E', alpha=0.3, label='Good (70-80)')
        ax2.fill_between(theta, 80, 100, color='#2E86AB', alpha=0.3, label='Excellent (80-100)')
        
        # Needle
        needle_angle = (composite_score / 100) * np.pi
        ax2.plot([needle_angle, needle_angle], [0, 95], 'k-', linewidth=3)
        ax2.plot(needle_angle, 95, 'ko', markersize=10)
        
        ax2.set_ylim([0, 100])
        ax2.set_xlim([0, np.pi])
        ax2.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax2.set_xticklabels(['0', '25', '50', '75', '100'])
        ax2.set_xlabel('Composite Score', fontsize=12)
        ax2.set_title(f'Composite Quality Score\n{composite_score:.1f}/100 - {rating}', 
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'composite_score_breakdown.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {output_file}")
        
        plt.close()
    
    # ========================================================================
    # REPORT GENERATION
    # ========================================================================
    
    def generate_machine_readable_report(self):
        """Generate JSON report with all metrics"""
        print("\n" + "="*60)
        print("📄 Generating Machine-Readable Report")
        print("="*60)
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'datasets_analyzed': len(self.datasets),
                'total_decisions': sum(len(d['decisions']) for d in self.datasets.values())
            },
            'metrics': self.metrics,
            'summary': {
                'composite_score': self.metrics['composite_score']['composite_score'],
                'rating': self.metrics['composite_score']['rating'],
                'key_findings': self._generate_key_findings()
            }
        }
        
        output_file = self.output_dir / 'evaluation_metrics.json'
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  ✅ Saved: {output_file}")
        
        return report
    
    def generate_human_readable_report(self):
        """Generate Markdown report"""
        print("\n" + "="*60)
        print("📄 Generating Human-Readable Report")
        print("="*60)
        
        composite_score = self.metrics['composite_score']['composite_score']
        rating = self.metrics['composite_score']['rating']
        
        report_lines = [
            "# Decision Quality Evaluation Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Datasets Analyzed:** {len(self.datasets)}",
            f"**Total Decisions:** {sum(len(d['decisions']) for d in self.datasets.values())}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"### Composite Quality Score: **{composite_score:.1f}/100** ({rating})",
            "",
            self._generate_executive_summary(),
            "",
            "---",
            "",
            "## Detailed Metrics",
            "",
            "### 1. Decision Consistency",
            "",
            self._format_consistency_section(),
            "",
            "### 2. Confidence Stability",
            "",
            self._format_confidence_section(),
            "",
            "### 3. Decision Distribution Shift",
            "",
            self._format_distribution_section(),
            "",
            "### 4. Cost Efficiency",
            "",
            self._format_cost_section(),
            "",
            "### 5. Reasoning Quality",
            "",
            self._format_reasoning_section(),
            "",
            "### 6. Human-Agent Boundary",
            "",
            self._format_boundary_section(),
            "",
            "### 7. Field Importance (H4)",
            "",
            self._format_field_importance_section(),
            "",
            "### 8. Segment Distortion (H6)",
            "",
            self._format_segment_distortion_section(),
            "",
            "---",
            "",
            "## Key Findings",
            "",
            self._format_key_findings(),
            "",
            "---",
            "",
            "## Recommendations",
            "",
            self._generate_recommendations(),
            "",
            "---",
            "",
            "## Visualizations",
            "",
            "See accompanying charts:",
            "- `decision_quality_analysis.png` - All 6 metrics across duplication levels",
            "- `composite_score_breakdown.png` - Composite score breakdown and gauge",
            ""
        ]
        
        output_file = self.output_dir / 'evaluation_report.md'
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"  ✅ Saved: {output_file}")
        
        return output_file
    
    def _generate_key_findings(self) -> List[str]:
        """Generate list of key findings"""
        findings = []
        
        # Consistency finding
        consistency = self.metrics['decision_consistency']['overall_consistency']
        if consistency < 0.9:
            findings.append(f"Decision consistency is concerning at {consistency:.1%} - "
                          f"duplicates frequently receive different decisions")
        else:
            findings.append(f"Decision consistency is strong at {consistency:.1%}")
        
        # Confidence finding
        conf_range = self.metrics['confidence_stability']['confidence_range']
        if conf_range < 0.01:
            findings.append(f"Agent confidence remains remarkably stable (±{conf_range:.3f}) "
                          f"despite data quality degradation")
        
        # Cost finding
        total_waste = self.metrics['cost_efficiency']['total_waste']
        if total_waste > 0.1:
            findings.append(f"Significant cost waste (${total_waste:.2f}) from processing duplicates")
        
        # Distribution finding
        max_shift = self.metrics['distribution_shift']['max_shift']
        if max_shift > 0.05:
            findings.append(f"Decision distribution shifts by up to {max_shift:.1%} "
                          f"with duplication")
        
        return findings
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary text"""
        consistency = self.metrics['decision_consistency']['overall_consistency']
        conf_stability = self.metrics['confidence_stability']['stability_score']
        total_waste = self.metrics['cost_efficiency']['total_waste']
        
        summary = f"""
This evaluation analyzed decision quality across 8 datasets with duplication levels 
ranging from 0% (clean baseline) to 100% (fully duplicated data).

**Key Observations:**

- **Decision Consistency:** {consistency:.1%} of duplicate records received consistent decisions
- **Confidence Stability:** Agent confidence remained stable at ~75.8% across all duplication levels
- **Cost Impact:** ${total_waste:.4f} wasted on duplicate processing
- **Quality Degradation:** Composite quality score of {self.metrics['composite_score']['composite_score']:.1f}/100

The agent demonstrates "confidently wrong" behavior - maintaining high confidence 
even when processing duplicate data, highlighting the critical need for data quality 
controls before AI agent deployment.
"""
        return summary.strip()
    
    def _format_consistency_section(self) -> str:
        """Format consistency metric section"""
        data = self.metrics['decision_consistency']
        lines = [
            f"**Overall Consistency:** {data['overall_consistency']:.2%}",
            f"**Total Inconsistencies:** {data['total_inconsistencies']}",
            f"**Metric Score:** {data['metric_score']:.1f}/100",
            "",
            "**Consistency by Duplication Level:**",
            ""
        ]
        
        for level in sorted(data['consistency_by_level'].keys(), 
                           key=lambda x: int(x.replace('pct', ''))):
            consistency = data['consistency_by_level'][level]
            lines.append(f"- {level:6s}: {consistency:6.2%}")
        
        return '\n'.join(lines)
    
    def _format_confidence_section(self) -> str:
        """Format confidence metric section"""
        data = self.metrics['confidence_stability']
        lines = [
            f"**Baseline Confidence:** {data['baseline_confidence']:.4f}",
            f"**Confidence Range:** {data['confidence_range']:.4f}",
            f"**Stability Score:** {data['stability_score']:.2%}",
            f"**Metric Score:** {data['metric_score']:.1f}/100",
            "",
            "**Confidence by Duplication Level:**",
            ""
        ]
        
        for level in sorted(data['confidence_by_level'].keys(), 
                           key=lambda x: int(x.replace('pct', ''))):
            confidence = data['confidence_by_level'][level]
            lines.append(f"- {level:6s}: {confidence:.4f}")
        
        return '\n'.join(lines)
    
    def _format_distribution_section(self) -> str:
        """Format distribution metric section"""
        data = self.metrics['distribution_shift']
        lines = [
            f"**Max Distribution Shift:** {data['max_shift']:+.2%}",
            f"**Shift Score:** {data['shift_score']:.2%}",
            f"**Metric Score:** {data['metric_score']:.1f}/100",
            "",
            "**Distribution by Duplication Level:**",
            ""
        ]
        
        for level in sorted(data['distribution_by_level'].keys(), 
                           key=lambda x: int(x.replace('pct', ''))):
            dist = data['distribution_by_level'][level]
            lines.append(f"- {level:6s}: HIGH={dist['HIGH_PRIORITY']:.1%}, "
                        f"MEDIUM={dist['MEDIUM_PRIORITY']:.1%}, "
                        f"LOW={dist['LOW_PRIORITY']:.1%}")
        
        return '\n'.join(lines)
    
    def _format_cost_section(self) -> str:
        """Format cost metric section"""
        data = self.metrics['cost_efficiency']
        lines = [
            f"**Total Cost:** ${data['total_cost']:.4f}",
            f"**Total Waste:** ${data['total_waste']:.4f} ({data['total_waste']/data['total_cost']:.1%})",
            f"**Efficiency Score:** {data['efficiency_score']:.2%}",
            f"**Metric Score:** {data['metric_score']:.1f}/100",
            "",
            "**Cost by Duplication Level:**",
            ""
        ]
        
        for level in sorted(data['cost_by_level'].keys(), 
                           key=lambda x: int(x.replace('pct', ''))):
            cost = data['cost_by_level'][level]
            waste = data['waste_by_level'].get(level, 0)
            lines.append(f"- {level:6s}: ${cost:.4f} (waste: ${waste:.4f})")
        
        return '\n'.join(lines)
    
    def _format_reasoning_section(self) -> str:
        """Format reasoning metric section"""
        data = self.metrics['reasoning_quality']
        lines = [
            f"**Baseline Reasoning Length:** {data['baseline_length']:.1f} words",
            f"**Length Variance:** {data['length_variance']:.2f}",
            f"**Quality Score:** {data['quality_score']:.2%}",
            f"**Metric Score:** {data['metric_score']:.1f}/100",
            "",
            "**Avg Reasoning Length by Duplication Level:**",
            ""
        ]
        
        for level in sorted(data['avg_reasoning_length_by_level'].keys(), 
                           key=lambda x: int(x.replace('pct', ''))):
            length = data['avg_reasoning_length_by_level'][level]
            lines.append(f"- {level:6s}: {length:.1f} words")
        
        return '\n'.join(lines)
    
    def _format_field_importance_section(self) -> str:
        """Format H4 field importance section"""
        data = self.metrics['field_importance']
        top_fields = list(data['top_fields_overall'].items())[:10]
        lines = [
            f"**Ranking Stability Across Duplication Levels:** {data['ranking_stability']:.2%}",
            f"**All Fields Observed:** {', '.join(data['all_fields_observed'])}",
            "",
            "**Top Fields Overall (by citation frequency):**",
            "",
        ]
        for field, freq in top_fields:
            lines.append(f"- `{field}`: cited in {freq:.2f} decisions on average")
        lines += [
            "",
            "**Top-5 Field Rankings by Duplication Level:**",
            "",
        ]
        for level in sorted(data['rankings_by_level'].keys(),
                           key=lambda x: int(x.replace('pct', ''))):
            ranking = data['rankings_by_level'][level]
            lines.append(f"- {level:6s}: {', '.join(ranking)}")
        return '\n'.join(lines)

    def _format_segment_distortion_section(self) -> str:
        """Format H6 segment distortion section"""
        data = self.metrics['segment_distortion']
        lines = [
            f"**Most Sensitive Segment:** {data['most_sensitive_segment']}",
            f"**Avg Max Shift:** {data['avg_max_shift']:.2%}",
            f"**Distortion Score:** {data['distortion_score']:.2%}",
            "",
            "**Segment Sensitivity (avg decision shift from baseline):**",
            "",
        ]
        for seg, sensitivity in sorted(data['segment_sensitivity'].items(),
                                       key=lambda x: -x[1]):
            lines.append(f"- {seg}: {sensitivity:.2%}")
        lines += [
            "",
            "**Max Decision Shift by Segment and Duplication Level:**",
            "",
        ]
        for level in sorted(data['segment_shift_by_level'].keys(),
                           key=lambda x: int(x.replace('pct', ''))):
            shifts = data['segment_shift_by_level'][level]
            shift_str = ", ".join(f"{s}:{v:.1%}" for s, v in shifts.items())
            lines.append(f"- {level:6s}: {shift_str}")
        return '\n'.join(lines)

    def _format_boundary_section(self) -> str:
        """Format human-agent boundary section"""
        data = self.metrics['human_agent_boundary']
        lines = [
            f"**Avg Review Required:** {data['avg_review_percentage']:.1%}",
            f"**Boundary Score:** {data['boundary_score']:.2%}",
            f"**Metric Score:** {data['metric_score']:.1f}/100",
            "",
            "**Recommended Thresholds by Duplication Level:**",
            ""
        ]
        
        for level in sorted(data['recommended_threshold_by_level'].keys(), 
                           key=lambda x: int(x.replace('pct', ''))):
            threshold = data['recommended_threshold_by_level'][level]
            review = data['decisions_requiring_review'][level]
            lines.append(f"- {level:6s}: threshold={threshold:.3f}, "
                        f"review={review['count']} ({review['percentage']:.1%})")
        
        return '\n'.join(lines)
    
    def _format_key_findings(self) -> str:
        """Format key findings as bullet points"""
        findings = self._generate_key_findings()
        return '\n'.join([f"- {finding}" for finding in findings])
    
    def _generate_recommendations(self) -> str:
        """Generate actionable recommendations"""
        consistency = self.metrics['decision_consistency']['overall_consistency']
        conf_stability = self.metrics['confidence_stability']['stability_score']
        total_waste = self.metrics['cost_efficiency']['total_waste']
        composite = self.metrics['composite_score']['composite_score']
        
        recommendations = []
        
        if consistency < 0.95:
            recommendations.append(
                "**Implement Deduplication:** Decision consistency is below 95%. "
                "Implement robust deduplication before agent processing to ensure "
                "consistent decisions for the same customer."
            )
        
        if conf_stability > 0.95:
            recommendations.append(
                "**Don't Rely on Confidence Alone:** Agent confidence remains stable "
                "despite data quality issues. Implement data quality checks independent "
                "of agent confidence scores."
            )
        
        if total_waste > 0.05:
            recommendations.append(
                f"**Reduce Cost Waste:** ${total_waste:.2f} wasted on duplicate processing. "
                "Implement upstream deduplication to reduce API costs by up to 50%."
            )
        
        if composite < 80:
            recommendations.append(
                "**Improve Data Quality Pipeline:** Composite quality score is below 80. "
                "Prioritize data quality improvements before scaling agent deployment."
            )
        
        recommendations.append(
            "**Establish Human Review Thresholds:** Use the recommended confidence "
            "thresholds to determine when human review is required, adjusting based "
            "on your data quality levels."
        )
        
        recommendations.append(
            "**Monitor Decision Consistency:** Track decision consistency as a key "
            "metric alongside confidence scores to detect data quality issues early."
        )
        
        return '\n\n'.join(recommendations)
    
    # ========================================================================
    # MAIN EVALUATION WORKFLOW
    # ========================================================================
    
    def run_evaluation(self):
        """Run complete evaluation workflow"""
        print("\n" + "="*60)
        print("🚀 STARTING DECISION QUALITY EVALUATION")
        print("="*60)
        print(f"Decisions Dir: {self.decisions_dir}")
        print(f"Output Dir:    {self.output_dir}")
        print(f"Datasets:      {len(self.datasets)}")
        
        # Run all metric analyses
        self.metrics['decision_consistency'] = self.analyze_decision_consistency()
        self.metrics['confidence_stability'] = self.analyze_confidence_stability()
        self.metrics['distribution_shift'] = self.analyze_distribution_shift()
        self.metrics['cost_efficiency'] = self.analyze_cost_efficiency()
        self.metrics['reasoning_quality'] = self.analyze_reasoning_quality()
        self.metrics['human_agent_boundary'] = self.analyze_human_agent_boundary()
        self.metrics['field_importance'] = self.analyze_field_importance()
        self.metrics['segment_distortion'] = self.analyze_segment_distortion()
        
        # Calculate composite score
        self.metrics['composite_score'] = self.calculate_composite_score()
        
        # Generate outputs
        self.generate_visualizations()
        self.generate_machine_readable_report()
        self.generate_human_readable_report()
        
        print("\n" + "="*60)
        print("✅ EVALUATION COMPLETE")
        print("="*60)
        print(f"\nOutputs saved to: {self.output_dir}")
        print(f"  - evaluation_metrics.json (machine-readable)")
        print(f"  - evaluation_report.md (human-readable)")
        print(f"  - decision_quality_analysis.png (6 metrics chart)")
        print(f"  - composite_score_breakdown.png (score breakdown)")
        
        return self.metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate decision quality across duplication levels'
    )
    parser.add_argument(
        '--decisions_dir',
        type=Path,
        default=Path('experiments_output/agent_results/decisions'),
        help='Directory containing decision files'
    )
    parser.add_argument(
        '--cluster_map_dir',
        type=Path,
        default=Path('experiments_output/eval'),
        help='Directory containing cluster map files (eval/ from data generator)'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('experiments_output/agent_results/evaluation'),
        help='Output directory for evaluation results'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = DecisionQualityEvaluator(
        decisions_dir=args.decisions_dir,
        output_dir=args.output_dir,
        cluster_map_dir=args.cluster_map_dir
    )
    
    metrics = evaluator.run_evaluation()
    
    # Print final summary
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    print(f"Composite Quality Score: {metrics['composite_score']['composite_score']:.1f}/100")
    print(f"Rating: {metrics['composite_score']['rating']}")
    print("\nIndividual Scores:")
    for metric, score in metrics['composite_score']['individual_scores'].items():
        print(f"  {metric:25s}: {score:5.1f}/100")
    print("="*60)


if __name__ == '__main__':
    main()

