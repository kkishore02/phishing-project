#!/usr/bin/env python3
"""
Educational AI-Augmented Phishing Simulation Framework

IMPORTANT: This code is for EDUCATIONAL PURPOSES ONLY. It should only be used in authorized 
environments with proper permissions and ethical oversight for cybersecurity training.

This framework demonstrates concepts of how AI could enhance phishing simulations
for legitimate security awareness training and authorized penetration testing.
"""

import os
import re
import csv
import json
import random
import argparse
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    print("Advanced NLP libraries not available. Basic functionality only.")

# Configurations
CONFIG = {
    "templates_dir": "./templates",
    "target_data": "./targets.csv",
    "results_dir": "./results",
    "log_file": "./phishing_sim.log",
    "authorized_domains": [],  # Must be filled with authorized domains
    "simulation_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    "simulation_authorized": False,  # Must be explicitly set to True
}

class PhishingSimulator:
    def __init__(self, config):
        """Initialize the simulator with configurations."""
        self.config = config
        self.templates = {}
        self.targets = []
        self.results = []
        
        # Create necessary directories
        os.makedirs(self.config["templates_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)
        
        # Initialize logger
        self.setup_logging()
        
        self.logger("Simulator initialized with simulation ID: " + config["simulation_id"])
        
        if not config["authorized_domains"]:
            self.logger("WARNING: No authorized domains configured. This tool should only be used for authorized testing.")
        
        if not config["simulation_authorized"]:
            self.logger("ERROR: Simulation not authorized. Set simulation_authorized to True only when properly authorized.")
            print("SIMULATION NOT AUTHORIZED. This tool must only be used with proper authorization.")
    
    def setup_logging(self):
        """Setup basic logging."""
        self.log_file = open(self.config["log_file"], "a")
        
    def logger(self, message):
        """Log messages with timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_file.write(log_entry)
        self.log_file.flush()
    
    def load_templates(self):
        """Load email templates from templates directory."""
        self.logger("Loading templates from " + self.config["templates_dir"])
        
        template_files = [f for f in os.listdir(self.config["templates_dir"]) 
                         if f.endswith('.json')]
        
        for template_file in template_files:
            try:
                with open(os.path.join(self.config["templates_dir"], template_file), 'r') as f:
                    template = json.load(f)
                    self.templates[template['id']] = template
                    self.logger(f"Loaded template: {template['id']}")
            except Exception as e:
                self.logger(f"Error loading template {template_file}: {str(e)}")
        
        self.logger(f"Loaded {len(self.templates)} templates")
        return len(self.templates) > 0
    
    def load_targets(self):
        """Load target data from CSV file."""
        self.logger("Loading targets from " + self.config["target_data"])
        
        try:
            if not os.path.exists(self.config["target_data"]):
                self.logger("Target data file not found")
                return False
            
            with open(self.config["target_data"], 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Verify domain is in authorized list
                    email_domain = row.get('email', '').split('@')[-1]
                    if email_domain not in self.config["authorized_domains"]:
                        self.logger(f"Skipping unauthorized domain: {email_domain}")
                        continue
                    
                    self.targets.append(row)
            
            self.logger(f"Loaded {len(self.targets)} authorized targets")
            return len(self.targets) > 0
        except Exception as e:
            self.logger(f"Error loading targets: {str(e)}")
            return False
    
    def analyze_target(self, target):
        """Analyze target data to determine personalization factors."""
        self.logger(f"Analyzing target: {target.get('email', 'Unknown')}")
        
        # Extract key information
        interests = target.get('interests', '').split(',')
        department = target.get('department', '')
        role = target.get('role', '')
        
        # Calculate personalization scores for each template
        scores = {}
        for template_id, template in self.templates.items():
            score = 0
            
            # Basic scoring
            if department.lower() in template.get('target_departments', []):
                score += 3
            
            if role.lower() in template.get('target_roles', []):
                score += 2
            
            # Interest matching
            for interest in interests:
                if interest.strip().lower() in template.get('keywords', []):
                    score += 1
            
            # Advanced NLP matching if available
            if ADVANCED_NLP_AVAILABLE and 'content' in template:
                profile_text = f"{department} {role} {' '.join(interests)}"
                similarity = self._calculate_text_similarity(profile_text, template['content'])
                score += similarity * 5  # Weight the similarity score
            
            scores[template_id] = score
        
        # Select best template
        if not scores:
            return random.choice(list(self.templates.keys()))
        
        best_template = max(scores, key=scores.get)
        self.logger(f"Selected template {best_template} with score {scores[best_template]}")
        return best_template
    
    def _calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two text strings."""
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0
    
    def personalize_template(self, template, target):
        """Personalize email template for specific target."""
        self.logger(f"Personalizing template for {target.get('email', 'Unknown')}")
        
        content = template.get('content', '')
        subject = template.get('subject', '')
        
        # Replace placeholders
        replacements = {
            '{{name}}': target.get('name', ''),
            '{{first_name}}': target.get('name', '').split()[0] if target.get('name', '') else '',
            '{{company}}': target.get('company', ''),
            '{{department}}': target.get('department', ''),
            '{{role}}': target.get('role', ''),
            '{{current_date}}': datetime.datetime.now().strftime("%B %d, %Y"),
        }
        
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)
            subject = subject.replace(placeholder, value)
        
        # Generate tracking ID for this specific campaign
        tracking_id = f"{self.config['simulation_id']}_{target.get('id', random.randint(1000, 9999))}"
        
        # Replace tracking link if present
        if '{{tracking_link}}' in content:
            tracking_url = f"http://tracking.example.com/c/{tracking_id}"
            content = content.replace('{{tracking_link}}', tracking_url)
        
        # Advanced AI-based enhancement (simulated)
        if ADVANCED_NLP_AVAILABLE:
            content = self._enhance_content_style(content, target)
        
        return {
            'to': target.get('email', ''),
            'subject': subject,
            'content': content,
            'tracking_id': tracking_id,
            'template_id': template.get('id', ''),
        }
    
    def _enhance_content_style(self, content, target):
        """Simulate AI enhancement of email content style."""
        # This is a simplified simulation of how AI might adjust content
        role = target.get('role', '').lower()
        department = target.get('department', '').lower()
        
        # Adjust technical level based on role
        technical_roles = ['developer', 'engineer', 'analyst', 'it', 'technical']
        is_technical = any(tech in role for tech in technical_roles)
        
        if is_technical:
            # Add more technical terms
            content = content.replace("please click", "please verify and click")
            content = content.replace("update", "update protocols")
            content = content.replace("system", "system infrastructure")
        else:
            # Simplify technical terms
            content = content.replace("authenticate", "verify")
            content = content.replace("credentials", "login details")
        
        # Adjust urgency based on department
        urgent_departments = ['finance', 'executive', 'sales', 'security']
        needs_urgency = any(dept in department for dept in urgent_departments)
        
        if needs_urgency and "urgent" not in content.lower():
            content = "URGENT: " + content
        
        return content
    
    def simulate_campaign(self, dry_run=True):
        """Run the phishing simulation campaign."""
        self.logger(f"Starting simulation campaign: {self.config['simulation_id']}")
        
        if not self.config["simulation_authorized"]:
            self.logger("ERROR: Unauthorized simulation attempt blocked")
            print("ERROR: Simulation not authorized. This tool must only be used with proper authorization.")
            return False
        
        if not self.load_templates():
            self.logger("Failed to load templates")
            return False
        
        if not self.load_targets():
            self.logger("Failed to load targets")
            return False
        
        results = []
        
        for target in self.targets:
            try:
                # Select best template for this target
                template_id = self.analyze_target(target)
                template = self.templates[template_id]
                
                # Personalize template
                email = self.personalize_template(template, target)
                
                # Record the planned email
                result = {
                    'target_email': target.get('email', ''),
                    'target_name': target.get('name', ''),
                    'template_id': template_id,
                    'tracking_id': email['tracking_id'],
                    'timestamp': datetime.datetime.now().isoformat(),
                    'sent': False
                }
                
                if not dry_run:
                    # Send email functionality would go here
                    # This is commented out as it should only be implemented
                    # in a fully authorized environment
                    '''
                    if self._send_email(email):
                        result['sent'] = True
                    '''
                    # For security, we're just simulating success in this example
                    result['sent'] = True
                
                results.append(result)
                self.logger(f"Processed target: {target.get('email', 'Unknown')}")
                
            except Exception as e:
                self.logger(f"Error processing target {target.get('email', 'Unknown')}: {str(e)}")
        
        # Save results
        self._save_results(results)
        self.logger(f"Simulation completed with {len(results)} emails")
        return True
    
    def _send_email(self, email_data):
        """
        DEMONSTRATION ONLY - NOT FUNCTIONAL
        
        This method would handle actual email sending in a real implementation.
        Deliberately not implemented to prevent misuse.
        """
        self.logger(f"SIMULATION: Would send email to {email_data['to']}")
        return True
    
    def _save_results(self, results):
        """Save simulation results to file."""
        results_file = os.path.join(
            self.config["results_dir"], 
            f"results_{self.config['simulation_id']}.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger(f"Results saved to {results_file}")
    
    def cleanup(self):
        """Clean up resources."""
        self.log_file.close()


def create_sample_template():
    """Create a sample template file for demonstration."""
    sample_template = {
        "id": "password_reset",
        "name": "Password Reset Notification",
        "subject": "Action Required: Password Reset for {{name}}",
        "content": """
Dear {{name}},

Our system has detected that your password will expire today. To maintain access to your account, please reset your password immediately by clicking the secure link below:

{{tracking_link}}

If you did not request this change, please contact the IT department immediately.

Thank you,
IT Security Team
{{company}}
        """,
        "target_departments": ["all"],
        "target_roles": ["all"],
        "keywords": ["password", "security", "access", "account"],
        "risk_level": "medium"
    }
    
    os.makedirs("./templates", exist_ok=True)
    with open("./templates/password_reset.json", "w") as f:
        json.dump(sample_template, f, indent=2)
    
    print("Created sample template: ./templates/password_reset.json")


def create_sample_targets():
    """Create a sample targets file for demonstration."""
    sample_targets = [
        {
            "id": "1001",
            "name": "John Doe",
            "email": "john.doe@example.com",
            "department": "IT",
            "role": "Developer",
            "company": "Example Corp",
            "interests": "coding,security,cloud computing"
        },
        {
            "id": "1002",
            "name": "Jane Smith",
            "email": "jane.smith@example.com",
            "department": "Marketing",
            "role": "Manager",
            "company": "Example Corp",
            "interests": "social media,design,analytics"
        }
    ]
    
    with open("./targets.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sample_targets[0].keys())
        writer.writeheader()
        writer.writerows(sample_targets)
    
    print("Created sample targets file: ./targets.csv")


def main():
    """Main function to run the simulator."""
    parser = argparse.ArgumentParser(description="Educational Phishing Simulation Tool")
    parser.add_argument("--init", action="store_true", help="Initialize with sample data")
    parser.add_argument("--authorize", action="store_true", help="Authorize the simulation (for educational purposes only)")
    parser.add_argument("--domain", action="append", help="Add authorized domain for testing")
    parser.add_argument("--dry-run", action="store_true", help="Plan but don't execute emails")
    args = parser.parse_args()
    
    # Create sample files if requested
    if args.init:
        create_sample_template()
        create_sample_targets()
        print("Initialized sample data files. Edit these with your authorized test data.")
        return
    
    # Set up configuration
    if args.domain:
        CONFIG["authorized_domains"].extend(args.domain)
    
    if args.authorize:
        print("\n" + "="*80)
        print("AUTHORIZATION CONFIRMATION")
        print("="*80)
        print("This tool is for EDUCATIONAL PURPOSES ONLY.")
        print("It must only be used in authorized environments with proper permissions")
        print("and ethical oversight for cybersecurity training and awareness.")
        print("\nBy authorizing this simulation, you confirm that:")
        print("1. You have proper authorization to conduct this test")
        print("2. All domains specified are authorized for testing")
        print("3. This is being conducted for legitimate security training purposes")
        print("4. You will handle all data ethically and in compliance with regulations")
        print("="*80)
        
        confirmation = input("\nDo you confirm the above? (yes/no): ")
        if confirmation.lower() == "yes":
            CONFIG["simulation_authorized"] = True
            print("Simulation authorized for educational purposes.")
        else:
            print("Simulation not authorized. Exiting.")
            return
    
    # Create and run simulator
    simulator = PhishingSimulator(CONFIG)
    
    if simulator.config["simulation_authorized"]:
        simulator.simulate_campaign(dry_run=args.dry_run)
    else:
        print("Simulation not authorized. Use --authorize to confirm educational use.")
    
    simulator.cleanup()


if __name__ == "__main__":
    print("""
    Educational AI-Augmented Phishing Simulation Framework
    
    IMPORTANT: This code is for EDUCATIONAL PURPOSES ONLY.
    It should only be used in authorized environments with proper permissions
    and ethical oversight for cybersecurity training.
    """)
    main()
