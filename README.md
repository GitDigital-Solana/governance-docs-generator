
### **governance-docs-generator/generate.py**
```python
#!/usr/bin/env python3
"""
Governance Documentation Generator
"""

import json
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import markdown
import jinja2
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet


class Format(Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"


@dataclass
class PolicyInfo:
    """Information about a policy for documentation."""
    name: str
    version: str
    description: str
    severity: str
    target_type: str
    rules_count: int
    created_at: str
    file_path: str
    content: Dict
    compliance_frameworks: List[str] = None


class DocumentationGenerator:
    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = Path(template_dir) if template_dir else Path(__file__).parent / "templates"
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def generate_from_directory(self, input_dir: str, output_dir: str, format: Format) -> Dict:
        """Generate documentation from a directory of policies."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Collect all policies
        policies = []
        for file_path in input_path.rglob("*.yaml"):
            policies.append(self._parse_policy(file_path))
        
        for file_path in input_path.rglob("*.yml"):
            policies.append(self._parse_policy(file_path))
        
        for file_path in input_path.rglob("*.json"):
            policies.append(self._parse_policy(file_path))
        
        # Generate documentation
        if format == Format.MARKDOWN:
            return self._generate_markdown(policies, output_path)
        elif format == Format.HTML:
            return self._generate_html(policies, output_path)
        elif format == Format.PDF:
            return self._generate_pdf(policies, output_path)
        elif format == Format.JSON:
            return self._generate_json(policies, output_path)
        elif format == Format.CSV:
            return self._generate_csv(policies, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _parse_policy(self, file_path: Path) -> PolicyInfo:
        """Parse a policy file into PolicyInfo."""
        with open(file_path, 'r') as f:
            if file_path.suffix in ['.yaml', '.yml']:
                content = yaml.safe_load(f)
            else:
                content = json.load(f)
        
        metadata = content.get('metadata', {})
        spec = content.get('spec', {})
        
        return PolicyInfo(
            name=metadata.get('name', 'Unknown'),
            version=metadata.get('version', '1.0.0'),
            description=metadata.get('description', ''),
            severity=metadata.get('severity', 'medium'),
            target_type=spec.get('target', {}).get('resourceType', 'unknown'),
            rules_count=len(spec.get('rules', [])),
            created_at=metadata.get('createdAt', datetime.now().isoformat()),
            file_path=str(file_path),
            content=content,
            compliance_frameworks=metadata.get('compliance', [])
        )
    
    def _generate_markdown(self, policies: List[PolicyInfo], output_path: Path) -> Dict:
        """Generate markdown documentation."""
        results = {}
        
        # Generate policy catalog
        catalog_path = output_path / "policy-catalog.md"
        with open(catalog_path, 'w') as f:
            f.write(self._render_markdown_catalog(policies))
        results['catalog'] = str(catalog_path)
        
        # Generate individual policy docs
        policy_docs_dir = output_path / "policies"
        policy_docs_dir.mkdir(exist_ok=True)
        
        for policy in policies:
            policy_path = policy_docs_dir / f"{policy.name}.md"
            with open(policy_path, 'w') as f:
                f.write(self._render_markdown_policy(policy))
            results[policy.name] = str(policy_path)
        
        # Generate compliance matrix
        matrix_path = output_path / "compliance-matrix.md"
        with open(matrix_path, 'w') as f:
            f.write(self._render_compliance_matrix(policies))
        results['matrix'] = str(matrix_path)
        
        return results
    
    def _render_markdown_catalog(self, policies: List[PolicyInfo]) -> str:
        """Render markdown catalog."""
        template = self.jinja_env.get_template("catalog.md.j2")
        return template.render(
            policies=policies,
            generated_at=datetime.now(),
            total_policies=len(policies),
            severity_counts=self._count_severities(policies)
        )
    
    def _render_markdown_policy(self, policy: PolicyInfo) -> str:
        """Render individual policy documentation."""
        template = self.jinja_env.get_template("policy.md.j2")
        return template.render(policy=policy)
    
    def _render_compliance_matrix(self, policies: List[PolicyInfo]) -> str:
        """Render compliance matrix."""
        # Extract compliance frameworks
        frameworks = set()
        for policy in policies:
            if policy.compliance_frameworks:
                frameworks.update(policy.compliance_frameworks)
        
        template = self.jinja_env.get_template("matrix.md.j2")
        return template.render(
            policies=policies,
            frameworks=sorted(frameworks),
            generated_at=datetime.now()
        )
    
    def _generate_html(self, policies: List[PolicyInfo], output_path: Path) -> Dict:
        """Generate HTML documentation."""
        results = {}
        
        # Generate main index
        index_path = output_path / "index.html"
        with open(index_path, 'w') as f:
            html = self._markdown_to_html(self._render_markdown_catalog(policies))
            f.write(self._wrap_html(html, "Policy Catalog"))
        results['index'] = str(index_path)
        
        # Generate policy pages
        policy_dir = output_path / "policies"
        policy_dir.mkdir(exist_ok=True)
        
        for policy in policies:
            policy_path = policy_dir / f"{policy.name}.html"
            with open(policy_path, 'w') as f:
                html = self._markdown_to_html(self._render_markdown_policy(policy))
                f.write(self._wrap_html(html, f"Policy: {policy.name}"))
            results[policy.name] = str(policy_path)
        
        return results
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert markdown to HTML."""
        return markdown.markdown(
            markdown_text,
            extensions=['tables', 'fenced_code', 'codehilite']
        )
    
    def _wrap_html(self, content: str, title: str) -> str:
        """Wrap content in HTML template."""
        template = self.jinja_env.get_template("base.html.j2")
        return template.render(title=title, content=content)
    
    def _generate_pdf(self, policies: List[PolicyInfo], output_path: Path) -> Dict:
        """Generate PDF documentation."""
        results = {}
        
        # Generate summary PDF
        summary_path = output_path / "policy-summary.pdf"
        self._create_pdf_summary(policies, summary_path)
        results['summary'] = str(summary_path)
        
        # Generate individual policy PDFs
        pdf_dir = output_path / "policies"
        pdf_dir.mkdir(exist_ok=True)
        
        for policy in policies:
            policy_path = pdf_dir / f"{policy.name}.pdf"
            self._create_pdf_policy(policy, policy_path)
            results[policy.name] = str(policy_path)
        
        return results
    
    def _create_pdf_summary(self, policies: List[PolicyInfo], output_path: Path):
        """Create PDF summary document."""
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("Governance Policy Catalog", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Summary
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"Total Policies: {len(policies)}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Table of policies
        data = [['Name', 'Version', 'Severity', 'Rules', 'Target']]
        for policy in policies:
            data.append([
                policy.name,
                policy.version,
                policy.severity,
                str(policy.rules_count),
                policy.target_type
            ])
        
        table = Table(data)
        story.append(table)
        
        doc.build(story)
    
    def _create_pdf_policy(self, policy: PolicyInfo, output_path: Path):
        """Create PDF for individual policy."""
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph(f"Policy: {policy.name}", styles['Title']))
        story.append(Paragraph(f"Version: {policy.version}", styles['Normal']))
        story.append(Paragraph(f"Severity: {policy.severity}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Description", styles['Heading2']))
        story.append(Paragraph(policy.description, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Rules section
        story.append(Paragraph("Rules", styles['Heading2']))
        for rule in policy.content.get('spec', {}).get('rules', []):
            story.append(Paragraph(f"• {rule.get('name', 'Unnamed')}", styles['Normal']))
            if 'description' in rule:
                story.append(Paragraph(f"  {rule['description']}", styles['Normal']))
        
        doc.build(story)
    
    def _generate_json(self, policies: List[PolicyInfo], output_path: Path) -> Dict:
        """Generate JSON documentation."""
        output_file = output_path / "policies.json"
        
        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_policies": len(policies),
                "format_version": "1.0.0"
            },
            "policies": [
                {
                    "name": policy.name,
                    "version": policy.version,
                    "description": policy.description,
                    "severity": policy.severity,
                    "target_type": policy.target_type,
                    "rules_count": policy.rules_count,
                    "compliance_frameworks": policy.compliance_frameworks,
                    "file_path": policy.file_path
                }
                for policy in policies
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return {"json": str(output_file)}
    
    def _generate_csv(self, policies: List[PolicyInfo], output_path: Path) -> Dict:
        """Generate CSV documentation."""
        output_file = output_path / "policies.csv"
        
        data = []
        for policy in policies:
            data.append({
                "name": policy.name,
                "version": policy.version,
                "description": policy.description,
                "severity": policy.severity,
                "target_type": policy.target_type,
                "rules_count": policy.rules_count,
                "created_at": policy.created_at,
                "file_path": policy.file_path
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        return {"csv": str(output_file)}
    
    def _count_severities(self, policies: List[PolicyInfo]) -> Dict[str, int]:
        """Count policies by severity."""
        counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for policy in policies:
            severity = policy.severity.lower()
            if severity in counts:
                counts[severity] += 1
        return counts


def main():
    """CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate governance documentation')
    parser.add_argument('--input', '-i', required=True, help='Input directory with policies')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--format', '-f', choices=['markdown', 'html', 'pdf', 'json', 'csv'],
                       default='markdown', help='Output format')
    parser.add_argument('--template-dir', help='Custom template directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    generator = DocumentationGenerator(args.template_dir)
    
    try:
        results = generator.generate_from_directory(
            args.input,
            args.output,
            Format(args.format)
        )
        
        if args.verbose:
            print("Generated files:")
            for key, path in results.items():
                print(f"  {key}: {path}")
        
        print(f"✅ Documentation generated successfully in {args.format.upper()} format")
        
    except Exception as e:
        print(f"❌ Error generating documentation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# governance-docs-generator
For are Governance and Compliance Teams a Governance Docs Generator repository 
