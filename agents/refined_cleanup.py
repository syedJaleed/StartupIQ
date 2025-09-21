#!/usr/bin/env python3
"""
Refined folder cleanup and deployment preparation
Removes unnecessary files and creates clean deployment package
"""

import os
import shutil
import zipfile
import json
from datetime import datetime
from pathlib import Path

def refined_cleanup_and_deploy():
    """Clean up folder for refined agent deployment"""
    
    # Define essential files to keep
    essential_files = [
        "refined_agent_v3.py",      # Main refined agent
        "requirements.txt",         # Dependencies
        "README.md",               # Documentation
        "fintrack_solutions.json", # Sample input for testing
        "config.yaml"              # Configuration (if exists)
    ]
    
    # Files to remove (outdated, temporary, or unnecessary)
    files_to_remove = [
        # Old agent versions
        "enhanced_agent_final.py",
        "startup_statistical_agent.py",
        "startup_analysis_agent.py",
        "enhanced_startup_agent.py",
        
        # Demo and development files
        "demo.py",
        "demo_enhanced.py",
        "test_agent.py",
        
        # Documentation (keep only README.md)
        "steps_in_detail.py",
        "steps.md", 
        "steps.txt",
        "enhancement_summary.json",
        "enhanced_setup_guide.md",
        "FINAL_IMPLEMENTATION_GUIDE.md",
        "modifications_summary.json",
        
        # Temporary and log files
        "startup_analysis.log",
        "temp_data.json",
        "log_20250918.log",
        
        # Output files (will be regenerated)
        "output.json",
        "output_ziniosa.json",
        
        # Old sample data
        "ziniosa_nithravya.json",
        "greenwave_solutions.json",  # Keep fintrack instead
        "input.json",  # Will use fintrack_solutions.json
        
        # Explainability files (move to logs)
        "explainability_demo_20250919_232252.json",
        "explainability_greenwave_solutions_20250919_235511.json",
        
        # Enhanced requirements
        "requirements_enhanced.txt",
        
        # Windows specific
        "run_analysis.bat",
        
        # Deployment files from previous versions
        "cleanup_and_deploy.py",
        "agent_deployment.zip",
        "MANIFEST.txt"
    ]
    
    # Directories to remove
    dirs_to_remove = [
        "__pycache__",
        "agent_deployment"
    ]
    
    removed_files = []
    retained_files = []
    
    print("🧹 Starting refined folder cleanup...")
    
    # Create logs directory for explainability files
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Move explainability files to logs directory
    explainability_files = list(Path(".").glob("explainability_*.json"))
    for exp_file in explainability_files:
        try:
            shutil.move(str(exp_file), logs_dir / exp_file.name)
            print(f"  📁 Moved to logs/: {exp_file.name}")
        except Exception as e:
            print(f"  ⚠️ Could not move {exp_file.name}: {e}")
    
    # Remove unnecessary files
    for file_name in files_to_remove:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                file_path.unlink()
                removed_files.append(file_name)
                print(f"  ❌ Removed: {file_name}")
            except Exception as e:
                print(f"  ⚠️ Could not remove {file_name}: {e}")
    
    # Remove unnecessary directories
    for dir_name in dirs_to_remove:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                removed_files.append(f"{dir_name}/")
                print(f"  ❌ Removed directory: {dir_name}/")
            except Exception as e:
                print(f"  ⚠️ Could not remove directory {dir_name}: {e}")
    
    # Check which essential files exist
    for file_name in essential_files:
        file_path = Path(file_name)
        if file_path.exists():
            retained_files.append(file_name)
            print(f"  ✅ Retained: {file_name}")
        else:
            print(f"  ⚠️ Essential file missing: {file_name}")
    
    # Add any other important files that should be retained
    for file_path in Path(".").iterdir():
        if (file_path.is_file() and 
            file_path.name not in essential_files and 
            file_path.name not in [f.split('/')[-1] for f in removed_files] and
            not file_path.name.startswith('.') and 
            file_path.suffix in ['.py', '.json', '.md', '.txt', '.yaml', '.yml']):
            retained_files.append(file_path.name)
            print(f"  ✅ Retained additional: {file_path.name}")
    
    # Create refined MANIFEST.txt
    manifest_content = {
        "deployment_info": {
            "created_at": datetime.now().isoformat(),
            "agent_version": "refined_v3.0",
            "description": "Refined Startup Statistical Analysis Agent with rounded outputs, capped projections, and enhanced sentiment analysis"
        },
        "retained_files": sorted(retained_files),
        "removed_files": sorted(removed_files),
        "file_descriptions": {
            "refined_agent_v3.py": "Main refined analysis agent with all improvements",
            "requirements.txt": "Python dependencies for the agent",
            "README.md": "Documentation and usage instructions",
            "fintrack_solutions.json": "Sample FinTrack Solutions input for testing",
            "config.yaml": "Configuration file (if present)"
        },
        "improvements": [
            "✅ All numerical outputs rounded to 2 decimal places",
            "✅ Projections capped at realistic values (revenue ≤1500%, profit ≤1000%)",
            "✅ 5-year projections only with decay model",
            "✅ Enhanced VADER sentiment analysis for growth narratives",
            "✅ Comprehensive explainability files with methodology",
            "✅ Clean deployment package excluding logs and temporary files"
        ],
        "deployment_instructions": [
            "1. Install dependencies: pip install -r requirements.txt",
            "2. Copy fintrack_solutions.json to input.json (or use your own data)",
            "3. Run analysis: python refined_agent_v3.py",
            "4. Check output.json for results",
            "5. Review explainability files in logs/ directory"
        ],
        "logs_directory": "logs/ - Contains explainability files and analysis logs"
    }
    
    with open("MANIFEST.txt", "w", encoding="utf-8") as f:
        json.dump(manifest_content, f, indent=2, ensure_ascii=False)
    
    retained_files.append("MANIFEST.txt")
    print(f"  ✅ Created: MANIFEST.txt")
    
    # Create deployment zip (excluding logs directory)
    zip_filename = "refined_agent_deployment.zip"
    
    print(f"\n📦 Creating refined deployment zip: {zip_filename}")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_name in retained_files:
            file_path = Path(file_name)
            if file_path.exists() and file_path.is_file():
                zipf.write(file_path, file_path.name)
                print(f"  📁 Added to zip: {file_name}")
    
    # Get zip file size
    zip_size = Path(zip_filename).stat().st_size / 1024  # KB
    
    print(f"\n🎉 Refined deployment preparation complete!")
    print(f"  • Files retained: {len(retained_files)}")
    print(f"  • Files removed: {len(removed_files)}")
    print(f"  • Deployment zip: {zip_filename} ({zip_size:.1f} KB)")
    print(f"  • Manifest file: MANIFEST.txt")
    print(f"  • Logs directory: logs/ (excluded from zip)")
    
    return {
        "retained_files": retained_files,
        "removed_files": removed_files,
        "zip_filename": zip_filename,
        "zip_size_kb": zip_size,
        "logs_directory": "logs/"
    }

if __name__ == "__main__":
    result = refined_cleanup_and_deploy()
    
    print(f"\n📊 Refined Cleanup Summary:")
    print(f"  • Total files retained: {len(result['retained_files'])}")
    print(f"  • Total files removed: {len(result['removed_files'])}")
    print(f"  • Deployment zip size: {result['zip_size_kb']:.1f} KB")
    print(f"  • Logs directory: {result['logs_directory']}")
    print(f"\n🚀 Ready for refined deployment!")