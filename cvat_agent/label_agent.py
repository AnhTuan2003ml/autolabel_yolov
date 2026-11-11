import json
import re
from pathlib import Path


class LabelDescriptionAgent:
    def __init__(self):
        self.aliases = {
            'vegetation': 'visible vegetation including trees, bushes, grass, plants, foliage, and leaves',
            'tree': 'a tree including its trunk, branches, leaves, and canopy',
            'trees': 'trees including trunks, branches, leaves, and canopies',
            'sky': 'the visible sky',
            'road': 'the paved roadway surface',
            'wall': 'a wall or solid vertical barrier',
            'fence': 'a fence or railing',
            'terrain': 'bare ground or terrain',
            'person': 'a person',
            'pedestrian': 'a person walking or standing in the scene',
            'car': 'a passenger car, sedan, SUV, taxi, or pickup',
            'truck': 'a large freight truck or lorry',
            'bus': 'a bus or coach',
            'train': 'a train or railway vehicle',
            'bicycle': 'a bicycle including its frame and wheels',
            'motorcycle': 'a motorcycle including its frame and wheels',
            'rider': 'a person riding a bicycle or motorcycle',
            'pole': 'a utility pole, street pole, or narrow vertical pole',
            'building': 'a building or house',
            'sidewalk': 'a sidewalk or pavement beside the road',
            'traffic light': 'a traffic light signal',
            'traffic sign': 'a roadside traffic sign',
        }

    def describe(self, label):
        normalized = label.strip().lower().replace('_', ' ')
        if normalized in self.aliases:
            subject = self.aliases[normalized]
        else:
            parts = [part for part in re.split(r'[/\s-]+', normalized) if part]
            subject = ' '.join(parts) or 'the visible target object'
            if normalized.startswith('lane '):
                subject = f'a road lane marking described as {subject[5:]}'
            elif normalized.startswith('area '):
                subject = f'a drivable road area described as {subject[5:]}'
            else:
                subject = f'a visible {subject}'
        return subject

    def build(self, labels, destination):
        descriptions = [{'label': label, 'prompt': self.describe(label)} for label in labels]
        destination.write_text(json.dumps(descriptions, ensure_ascii=False, indent=2), encoding='utf-8')
        return descriptions
