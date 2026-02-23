# Recipes

JSON files that define tool requirements and coaching rules for each training scenario.

## Structure

- **aliases**: maps model class names to recipe names (e.g. `needle_driver` -> `needle_holder`)
- **preop_required**: tools that must be visible before practice starts
- **params**: detection confidence threshold, stability hold time
- **intraop_rules**: phase-specific rules with debounce timing

## Adding a new scenario

Copy `trauma_room.json`, change the tool list and rules. Tool names must match YOLO model classes (lowercase, underscores).
