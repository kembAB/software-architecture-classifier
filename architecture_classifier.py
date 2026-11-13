#!/usr/bin/env python3
"""
ArchitectureScope - Software Architecture Style Classifier
==========================================================

Classifies a software system into one of six architecture styles from
eight measurable structural features, using a k-nearest-neighbours
classifier built from scratch. Standard library only: no dependencies,
no API keys, no network access.

Run it with:

    python architecture_classifier.py

The program prints three things:
  1. A summary of the labelled dataset of 36 example systems.
  2. A leave-one-out cross-validation evaluation of the classifier.
  3. Live classifications of five new systems that are not in the
     dataset, each with its three nearest neighbours as evidence.

Author: Abel Haile Magicho
"""

import math

# ----------------------------------------------------------------------
# The eight structural features
# ----------------------------------------------------------------------
# Every system, in the dataset and in the demo, is described by the same
# eight numbers, always in this order:
#
#   1. deployment_units
#      How many separately deployed units the system has. A single
#      deployable application scores 1; a platform of many services
#      scores high.
#   2. independent_deployment
#      Ratio from 0.0 to 1.0 of units that can be deployed without
#      touching the other units.
#   3. own_database_per_unit
#      1 when each unit owns its own data store, 0 when units share
#      one database.
#   4. async_communication_ratio
#      Ratio from 0.0 to 1.0 of communication between units that is
#      asynchronous messaging rather than direct synchronous calls.
#   5. coupling_score
#      0 to 10. Higher means tighter coupling between units: a change
#      in one unit is more likely to force changes in others.
#   6. sequential_pipeline_flow
#      1 when data mainly flows through a sequence of transformation
#      stages, 0 otherwise.
#   7. client_request_driven
#      1 when work is mainly initiated by client requests directed at
#      a server side, 0 when work is mainly initiated by events or by
#      data arriving at the head of a pipeline.
#   8. team_autonomy
#      0 to 10. How independently separate teams own separate units.

FEATURE_NAMES = [
    "deployment_units",
    "independent_deployment",
    "own_database_per_unit",
    "async_communication_ratio",
    "coupling_score",
    "sequential_pipeline_flow",
    "client_request_driven",
    "team_autonomy",
]

STYLES = [
    "Monolith",
    "Layered",
    "Client-Server",
    "Microservices",
    "Event-Driven",
    "Pipe-and-Filter",
]

# ----------------------------------------------------------------------
# The labelled dataset: 36 example systems, 6 per style.
# Each entry is a name, the true style, and the eight feature values in
# the order defined above. The values are hand-crafted from the
# published characteristics of each style, with realistic variation
# inside each style.
# ----------------------------------------------------------------------

DATASET = [
    # --- Monolith: one deployable unit, shared database, tight coupling
    {"name": "SmallShop web store", "style": "Monolith",
     "features": (1, 0.0, 0, 0.05, 9, 0, 1, 2)},
    {"name": "ClinicDesk records application", "style": "Monolith",
     "features": (1, 0.0, 0, 0.0, 8, 0, 1, 1)},
    {"name": "Legacy payroll application", "style": "Monolith",
     "features": (1, 0.0, 0, 0.1, 9, 0, 1, 2)},
    {"name": "University course portal", "style": "Monolith",
     "features": (1, 0.0, 0, 0.0, 8, 0, 1, 3)},
    {"name": "GarageWorks workshop manager", "style": "Monolith",
     "features": (1, 0.0, 0, 0.05, 10, 0, 1, 1)},
    {"name": "Hotel booking application", "style": "Monolith",
     "features": (1, 0.0, 0, 0.1, 8, 0, 1, 2)},

    # --- Layered: a few tiers deployed together, shared database
    {"name": "Bank teller three-tier system", "style": "Layered",
     "features": (3, 0.1, 0, 0.05, 6, 0, 1, 4)},
    {"name": "Insurance claims layered platform", "style": "Layered",
     "features": (3, 0.2, 0, 0.1, 6, 0, 1, 5)},
    {"name": "Hospital records n-tier application", "style": "Layered",
     "features": (2, 0.0, 0, 0.0, 5, 0, 1, 3)},
    {"name": "Government permits tiered portal", "style": "Layered",
     "features": (4, 0.1, 0, 0.05, 6, 0, 1, 4)},
    {"name": "Retail inventory layered suite", "style": "Layered",
     "features": (2, 0.2, 0, 0.1, 5, 0, 1, 5)},
    {"name": "Airline check-in tiered system", "style": "Layered",
     "features": (3, 0.0, 0, 0.0, 6, 0, 1, 3)},

    # --- Client-Server: many client units around one server side
    {"name": "Email system with desktop clients", "style": "Client-Server",
     "features": (8, 0.4, 0, 0.2, 4, 0, 1, 4)},
    {"name": "Multiplayer game client and server", "style": "Client-Server",
     "features": (12, 0.5, 0, 0.3, 4, 0, 1, 5)},
    {"name": "Point-of-sale terminals with store server",
     "style": "Client-Server",
     "features": (6, 0.4, 0, 0.1, 5, 0, 1, 3)},
    {"name": "Library catalogue terminals", "style": "Client-Server",
     "features": (5, 0.3, 0, 0.05, 4, 0, 1, 3)},
    {"name": "Call-centre desktops and server", "style": "Client-Server",
     "features": (10, 0.5, 0, 0.15, 3, 0, 1, 5)},
    {"name": "Warehouse scanner clients", "style": "Client-Server",
     "features": (7, 0.4, 0, 0.2, 5, 0, 1, 4)},

    # --- Microservices: many independent services, own databases
    {"name": "E-commerce platform of 24 services", "style": "Microservices",
     "features": (24, 0.9, 1, 0.4, 3, 0, 1, 9)},
    {"name": "Ride-hailing backend services", "style": "Microservices",
     "features": (35, 1.0, 1, 0.5, 2, 0, 1, 10)},
    {"name": "Streaming media service mesh", "style": "Microservices",
     "features": (40, 0.9, 1, 0.3, 3, 0, 1, 9)},
    {"name": "Fintech payments microservices", "style": "Microservices",
     "features": (18, 0.8, 1, 0.4, 4, 0, 1, 8)},
    {"name": "Travel booking microservices", "style": "Microservices",
     "features": (15, 0.9, 1, 0.2, 3, 0, 1, 8)},
    {"name": "Food delivery platform services", "style": "Microservices",
     "features": (28, 1.0, 1, 0.6, 2, 0, 1, 10)},

    # --- Event-Driven: work starts from events, asynchronous throughout
    {"name": "IoT sensor event backbone", "style": "Event-Driven",
     "features": (20, 0.9, 1, 0.95, 2, 0, 0, 8)},
    {"name": "Fraud detection event platform", "style": "Event-Driven",
     "features": (12, 0.8, 1, 0.9, 2, 0, 0, 9)},
    {"name": "Logistics tracking event system", "style": "Event-Driven",
     "features": (16, 0.9, 0, 0.85, 3, 0, 0, 7)},
    {"name": "Smart building event hub", "style": "Event-Driven",
     "features": (9, 0.7, 1, 0.8, 2, 0, 0, 8)},
    {"name": "Stock trade event processing platform", "style": "Event-Driven",
     "features": (25, 1.0, 1, 1.0, 1, 0, 0, 10)},
    {"name": "Social notification event system", "style": "Event-Driven",
     "features": (14, 0.8, 1, 0.9, 3, 0, 0, 7)},

    # --- Pipe-and-Filter: data flows through transformation stages
    {"name": "Image processing filter chain", "style": "Pipe-and-Filter",
     "features": (5, 0.7, 0, 0.5, 3, 1, 0, 5)},
    {"name": "Compiler stage chain", "style": "Pipe-and-Filter",
     "features": (6, 0.8, 0, 0.3, 2, 1, 0, 6)},
    {"name": "Video transcoding pipeline", "style": "Pipe-and-Filter",
     "features": (7, 0.6, 0, 0.6, 3, 1, 0, 5)},
    {"name": "Nightly ETL data pipeline", "style": "Pipe-and-Filter",
     "features": (4, 0.5, 0, 0.4, 4, 1, 0, 4)},
    {"name": "Audio mastering filter stages", "style": "Pipe-and-Filter",
     "features": (5, 0.9, 0, 0.5, 2, 1, 0, 6)},
    {"name": "Document conversion pipeline", "style": "Pipe-and-Filter",
     "features": (8, 0.7, 0, 0.7, 3, 1, 0, 7)},
]

# ----------------------------------------------------------------------
# Five new systems for the live demo. None of them is in the dataset.
# ----------------------------------------------------------------------

DEMO_SYSTEMS = [
    {"name": "Harbour Books online store",
     "description": "A small online bookstore built and deployed as one "
                    "single application with one shared database.",
     "features": (1, 0.0, 0, 0.0, 8, 0, 1, 2)},
    {"name": "Continental Bank service platform",
     "description": "A banking platform of thirty independently deployed "
                    "services, each owning its database, mixing direct "
                    "calls with messaging.",
     "features": (30, 0.95, 1, 0.45, 3, 0, 1, 9)},
    {"name": "Factory telemetry monitoring system",
     "description": "Sensors publish events to a backbone where "
                    "independent processors react to them.",
     "features": (18, 0.9, 1, 0.95, 2, 0, 0, 8)},
    {"name": "PhotoLab enhancement tool",
     "description": "A photo tool that chains resize, denoise, colour "
                    "correction and export stages in sequence.",
     "features": (5, 0.75, 0, 0.45, 3, 1, 0, 5)},
    {"name": "Dental clinic reception system",
     "description": "Reception desktops at several clinics all talk to "
                    "one central server holding the records.",
     "features": (6, 0.4, 0, 0.1, 4, 0, 1, 3)},
]

K = 3  # number of neighbours that vote on every classification


# ----------------------------------------------------------------------
# Min-max normalization
# ----------------------------------------------------------------------

def fit_min_max(systems):
    """Return the per-feature minimum and maximum over a list of systems."""
    columns = list(zip(*(s["features"] for s in systems)))
    mins = [min(col) for col in columns]
    maxs = [max(col) for col in columns]
    return mins, maxs


def normalize(features, mins, maxs):
    """Scale each feature into the range 0.0 to 1.0."""
    scaled = []
    for value, low, high in zip(features, mins, maxs):
        if high == low:
            scaled.append(0.0)
        else:
            scaled.append((value - low) / (high - low))
    return tuple(scaled)


# ----------------------------------------------------------------------
# The k-nearest-neighbours classifier
# ----------------------------------------------------------------------

def euclidean(first, second):
    """Straight-line distance between two normalized feature vectors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def classify(features, reference_systems, k=K):
    """
    Classify one system against a list of labelled reference systems.

    The normalization is fitted on the reference systems. Returns the
    predicted style and the k nearest neighbours, each as a tuple of
    distance, name and true style, sorted by distance. Voting is by
    majority; ties are broken by the smaller total distance, then by
    the single nearest neighbour.
    """
    mins, maxs = fit_min_max(reference_systems)
    target = normalize(features, mins, maxs)
    scored = []
    for system in reference_systems:
        candidate = normalize(system["features"], mins, maxs)
        scored.append((euclidean(target, candidate),
                       system["name"], system["style"]))
    scored.sort(key=lambda item: item[0])
    neighbours = scored[:k]

    votes = {}
    distance_sums = {}
    for distance, _name, style in neighbours:
        votes[style] = votes.get(style, 0) + 1
        distance_sums[style] = distance_sums.get(style, 0.0) + distance
    best_votes = max(votes.values())
    leaders = [style for style, count in votes.items() if count == best_votes]
    if len(leaders) == 1:
        prediction = leaders[0]
    else:
        prediction = min(leaders, key=lambda style: distance_sums[style])
    return prediction, neighbours


# ----------------------------------------------------------------------
# Evaluation: leave-one-out cross-validation
# ----------------------------------------------------------------------

def leave_one_out_evaluation(dataset):
    """
    Classify every system in the dataset using all the other systems
    as the reference set, and compare each prediction with the truth.
    Returns the number correct, per-style tallies and the list of
    misclassifications.
    """
    per_style_total = {style: 0 for style in STYLES}
    per_style_correct = {style: 0 for style in STYLES}
    mistakes = []
    correct = 0
    for index, system in enumerate(dataset):
        reference = dataset[:index] + dataset[index + 1:]
        prediction, neighbours = classify(system["features"], reference)
        per_style_total[system["style"]] += 1
        if prediction == system["style"]:
            correct += 1
            per_style_correct[system["style"]] += 1
        else:
            mistakes.append((system, prediction, neighbours))
    return correct, per_style_correct, per_style_total, mistakes


# ----------------------------------------------------------------------
# Printing helpers
# ----------------------------------------------------------------------

def format_features(features):
    """Render the eight feature values as one readable line."""
    parts = []
    for name, value in zip(FEATURE_NAMES, features):
        if isinstance(value, float):
            parts.append("%s=%.2f" % (name, value))
        else:
            parts.append("%s=%s" % (name, value))
    return ", ".join(parts)


def print_neighbour_evidence(neighbours):
    for distance, name, style in neighbours:
        print("      - %-42s %-15s distance %.3f" % (name, style, distance))


# ----------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ArchitectureScope - Software Architecture Style Classifier")
    print("=" * 70)
    print()
    print("Classifies a software system into one of six architecture")
    print("styles from eight measurable structural features, using a")
    print("k-nearest-neighbours classifier with k = %d." % K)
    print()

    print("-" * 70)
    print("DATASET")
    print("-" * 70)
    print("%d labelled example systems, %d per style."
          % (len(DATASET), len(DATASET) // len(STYLES)))
    print("Styles: " + ", ".join(sorted(STYLES)))
    print("Every system is measured on the same eight features:")
    for position, name in enumerate(FEATURE_NAMES, start=1):
        print("  %d. %s" % (position, name))
    print()

    print("-" * 70)
    print("EVALUATION - LEAVE-ONE-OUT CROSS-VALIDATION")
    print("-" * 70)
    correct, per_style_correct, per_style_total, mistakes = \
        leave_one_out_evaluation(DATASET)
    total = len(DATASET)
    accuracy = 100.0 * correct / total
    print("Each of the %d systems was classified using the other %d as"
          % (total, total - 1))
    print("the reference set, and the prediction was checked against")
    print("the true style.")
    print()
    print("Overall accuracy: %d/%d = %.1f%%" % (correct, total, accuracy))
    print()
    print("Per-style results:")
    for style in sorted(STYLES):
        print("  %-15s %d/%d correct"
              % (style, per_style_correct[style], per_style_total[style]))
    print()
    if mistakes:
        print("Misclassifications: %d" % len(mistakes))
        for system, prediction, neighbours in mistakes:
            print("  * %s" % system["name"])
            print("      true style: %s" % system["style"])
            print("      predicted : %s" % prediction)
            print("      features  : %s" % format_features(system["features"]))
            print("      nearest neighbours:")
            print_neighbour_evidence(neighbours)
    else:
        print("Misclassifications: none. Every system was matched to")
        print("its true style by its three nearest neighbours.")
    print()

    print("-" * 70)
    print("DEMO - FIVE NEW SYSTEMS CLASSIFIED LIVE")
    print("-" * 70)
    print("None of these systems is in the dataset. Each one is")
    print("classified against all %d labelled examples." % len(DATASET))
    print()
    for position, system in enumerate(DEMO_SYSTEMS, start=1):
        prediction, neighbours = classify(system["features"], DATASET)
        print("%d. %s" % (position, system["name"]))
        print("   %s" % system["description"])
        print("   Features: %s" % format_features(system["features"]))
        print("   Predicted style: %s" % prediction)
        print("   Nearest neighbours:")
        print_neighbour_evidence(neighbours)
        print()


if __name__ == "__main__":
    main()
