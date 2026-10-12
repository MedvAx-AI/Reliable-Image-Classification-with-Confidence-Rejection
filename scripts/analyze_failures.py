"""Save three actual paired failure examples and evidence-grounded interpretation."""
import csv
import gzip
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

from src.corruptions import corruption_specs, image_sha256
from src.data import Cifar10SplitDataset
from src.failures import select_failures
from src.policy_artifacts import load_frozen_policy, read_json, write_json


def analyze():
    policy = load_frozen_policy()
    with gzip.open(ROOT / 'results/per_image.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ('accepted', 'correct'):
            row[key] = row[key] == 'True'
        for key in ('raw_confidence', 'calibrated_confidence'):
            row[key] = float(row[key])
        for key in ('true_class', 'predicted_class'):
            row[key] = int(row[key])
        row['active_threshold'] = float(row['active_threshold']) if row['active_threshold'] else None
    result = select_failures(rows)
    if len(result['cases']) < 3:
        raise RuntimeError(f"Not three available categories: {result['missing_categories']}; do not invent cases")
    examples = ROOT / 'results/failures'
    examples.mkdir(parents=True, exist_ok=True)
    clean_ds = Cifar10SplitDataset('final_test')
    hypotheses = {
        'confident_wrong_accepted': ('The fixed ImageNet features and linear CIFAR head may rely on ambiguous shape/background cues in this tiny image. High softmax confidence measures relative class preference and does not prove correctness.',
                                    'Compare foreground/background masking and nearest development features with the checkpoint fixed; a selected image alone cannot reveal internal model reasoning.'),
        'correct_but_rejected': ('The correct class has insufficient separation from competing logits to pass the clean-fitted threshold. Rejection can discard correct predictions as well as errors.',
                                'Inspect the top class margins and a larger set of correct rejections; lowering the threshold would change coverage and must not be tuned on this test example.'),
        'corruption_damage': ('The native-resolution corruption alters visual evidence before the same ImageNet resize/crop. The paired class/decision change supports sensitivity to lost or altered cues, but not a specific causal explanation inside the network.',
                              'Repeat the paired analysis on development images across severities; isolate foreground detail and record confidence transitions without changing the locked policy.'),
    }
    titles = {'confident_wrong_accepted':'Confidently wrong and accepted',
              'correct_but_rejected':'Correct latent class, unnecessarily rejected',
              'corruption_damage':'Correct clean decision becomes a wrong accepted decision'}
    observed_hypotheses = {
        'test:5806': 'The airplane has a small horizontal silhouette against a watery-looking background. Ambiguous shape and background cues at 32x32 may favor ship in the frozen features. This is a visual hypothesis, not proof of internal model reasoning.',
        'test:2998': 'The tightly cropped frontal dog face lacks body context. The predicted dog remains correct, but calibrated confidence 0.568 is just below the frozen 0.57 threshold. This demonstrates the cost of rejection; the threshold was not changed after seeing the image.',
        'test:4720': 'The small diagonal ship and its wake on blue water lose outline detail and gain visible block/band artifacts under JPEG quality 15. The same source changes from a correct accepted ship to a confidently wrong accepted airplane. Altered shape/wake cues are a plausible cause, not a demonstrated internal mechanism.',
    }
    font_path = Path('C:/Windows/Fonts/calibri.ttf')
    font = ImageFont.truetype(str(font_path), 21) if font_path.exists() else ImageFont.load_default()
    panel = Image.new('RGB', (1120, 3 * 255), '#F4F7FA')
    draw = ImageDraw.Draw(panel)
    report = ['# Final failure analysis', '',
              'These cases are selected from the frozen final run after evaluation. They explain observed limitations; they did not change T, thresholds, checkpoint or corruption settings.', '',
              f"Run: `{read_json(ROOT / 'results/run_manifest.json')['run_id']}`. T={policy['temperature']:.8f}; raw tau={policy['raw']['threshold']:.2f}, calibrated tau={policy['calibrated']['threshold']:.2f}.", '',
              '## Selection and counts', '', result['selection'] + '.', '',
              'Counts are occurrences in condition D. Clean categories each have denominator 2,000; corruption damage considers 6,000 image variants (three severe corruptions), with paired clean D decisions. Counts of unique source images are also reported.', '']
    for category, counts in result['category_counts'].items():
        report.append(f"- {category}: {counts['candidate_rows']} candidate rows / {counts['unique_source_images']} unique source images.")
    specs = {(s['corruption'], s['severity']): s for s in corruption_specs()}
    for index, case in enumerate(result['cases']):
        row, clean = case['row'], case['clean']
        spec = specs[(row['corruption'], row['severity'])]
        clean_image = clean_ds.get_by_source_id(row['source_id']).image
        if row['corruption'] == 'clean':
            observed = clean_image
        else:
            observed = Cifar10SplitDataset('final_test', corruption=spec['corruption'], severity=spec['severity'],
                                         param_value=spec['param_value']).get_by_source_id(row['source_id']).image
        Image.fromarray(clean_image).save(examples / f'case_{index+1}_clean.png')
        Image.fromarray(observed).save(examples / f'case_{index+1}_observed.png')
        case.update(title=titles[case['category']], hypothesis=observed_hypotheses.get(row['source_id'], hypotheses[case['category']][0]),
                    limitation_and_next_test=hypotheses[case['category']][1],
                    corruption_parameters=spec, clean_image_sha256=image_sha256(clean_image),
                    observed_image_sha256=image_sha256(observed))
        y = index * 255
        panel.paste(Image.fromarray(clean_image).resize((176, 176), Image.Resampling.NEAREST), (18, y+45))
        panel.paste(Image.fromarray(observed).resize((176, 176), Image.Resampling.NEAREST), (210, y+45))
        draw.text((18,y+12),'Clean',font=font,fill='#162B40')
        draw.text((210,y+12),f"{row['corruption']}/{row['severity']}",font=font,fill='#162B40')
        truth = policy['class_order'][row['true_class']]
        prediction = policy['class_order'][row['predicted_class']]
        text = [f"{index+1}. {titles[case['category']]}", f"{row['source_id']} | true: {truth} | predicted: {prediction}",
                f"Raw confidence {row['raw_confidence']:.3f}; calibrated {row['calibrated_confidence']:.3f}",
                f"D threshold {row['active_threshold']:.2f}; {'ACCEPT' if row['accepted'] else 'REJECT'}",
                f"Clean: {policy['class_order'][clean['predicted_class']]} / confidence {clean['calibrated_confidence']:.3f}"]
        for j,line in enumerate(text):
            draw.text((412,y+24+j*35),line,font=font,fill='#162B40')
        report.extend(['', f"## {index+1}. {titles[case['category']]}", '',
                       f"![Clean view](../results/failures/case_{index+1}_clean.png) ![Observed view](../results/failures/case_{index+1}_observed.png)", '',
                       f"Source **{row['source_id']}**, final_test; true **{truth}**, observed prediction **{prediction}**. Condition **{row['corruption']}/{row['severity']}**, parameter {spec['param_value']}. Raw confidence **{row['raw_confidence']:.6f}**, calibrated **{row['calibrated_confidence']:.6f}**, D threshold **{row['active_threshold']:.2f}**, decision **{'ACCEPT' if row['accepted'] else 'REJECT'}**.", '',
                       f"Paired clean prediction: **{policy['class_order'][clean['predicted_class']]}**, raw confidence {clean['raw_confidence']:.6f}, calibrated {clean['calibrated_confidence']:.6f}; {'ACCEPT' if clean['accepted'] else 'REJECT'}.", '',
                       '**Likely cause / hypothesis:** ' + case['hypothesis'], '',
                       '**Limitation / next test:** ' + case['limitation_and_next_test'], '',
                       f"Image hashes: clean `{case['clean_image_sha256']}`, observed `{case['observed_image_sha256']}`."])
    panel.save(ROOT / 'results/figures/failures.png')
    result['policy_sha256'] = read_json(ROOT / 'results/run_manifest.json')['policy_sha256']
    write_json(examples / 'cases.json', result)
    (ROOT / 'reports/failure_analysis.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('Saved three real cases, paired images, population counts and explanations.')


if __name__ == '__main__':
    analyze()
