import json
import os
import re
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable

import torch
from rich.console import Console
from transformers import pipeline, logging as hf_logging
from sentence_transformers import SentenceTransformer, util

hf_logging.set_verbosity_error()
console = Console()

@dataclass
class MirrorMindConfig:
    model_name: str = "google/flan-t5-large"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[int] = None

    max_iters: int = 4
    num_candidates: int = 3
    
    similarity_threshold: float = 0.99 
    critique_threshold: float = 8.8
    initial_temp: float = 0.8
    final_temp: float = 0.1

    output_dir: str = "mirror_sessions"

class TextGenerator:
    def __init__(self, model_name: str, device: Optional[int] = None):
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        
        console.log(f"Loading LLM: [cyan]{model_name}[/cyan] on device {device}")
        try:
            self.pipe = pipeline(
                "text2text-generation",
                model=model_name,
                tokenizer=model_name,
                device=device,
                max_length=512
            )
        except Exception as e:
            console.log(f"[red]Model Load Error:[/red] {e}")
            raise e

    def generate(self, prompt: str, temperature: float = 0.1, num_return: int = 1) -> List[str]:
        """Generates one or multiple candidates."""
        safe_temp = max(0.01, min(1.0, temperature))
        do_sample = safe_temp > 0.1
        
        try:
            outs = self.pipe(
                prompt,
                do_sample=do_sample,
                temperature=safe_temp,
                top_p=0.92,
                num_return_sequences=num_return,
                truncation=True
            )
            return [o["generated_text"].strip() for o in outs]
        except Exception as e:
            console.log(f"[red]Gen Error:[/red] {e}")
            return ["Error generation"] * num_return
            
class Embedder:
    def __init__(self, model_name: str, device: Optional[int] = None):
        dev = "cuda" if (device is not None and torch.cuda.is_available()) else "cpu"
        console.log(f"Loading Embedder: [cyan]{model_name}[/cyan] on {dev}")
        self.model = SentenceTransformer(model_name, device=dev)

    def get_score(self, source: str, candidate: str) -> float:
        embs = self.model.encode([source, candidate], convert_to_tensor=True)
        return float(util.cos_sim(embs[0], embs[1]).item())

class CriticPersona:
    def __init__(self, name: str, focus: str, prompt_behavior: str):
        self.name = name
        self.focus = focus
        self.prompt_behavior = prompt_behavior

class CriticEnsemble:
    def __init__(self, generator: TextGenerator):
        self.generator = generator
        self.personas = [
            CriticPersona("The Logician", "Logical Consistency", "Find contradictions, non-sequiturs, and counting/math errors."),
            CriticPersona("The Skeptic", "Factuality & Evidence", "Assume the text is lying. Demand proof/precision."),
            CriticPersona("The Editor", "Clarity & Style", "Check for repetition, vague words, and formatting.")
        ]

    def _parse_json_robust(self, text: str, persona: str) -> Dict[str, Any]:
        """Unbreakable JSON extraction and normalization."""
        text = text.strip()
        default_score = 5
        match = re.search(r'\{.*\}', text, re.DOTALL)
        json_cand = match.group(0) if match else text
        
        score = default_score
        critique = "Could not parse detailed critique text."
        
        try:
            data = json.loads(json_cand)
            score = int(data.get("Score", default_score))
            critique = data.get("Critique", "No text provided in JSON.")
        except:
            s_match = re.search(r'"?Score"?\s*[:=]\s*(\d+)', text)
            if s_match: score = int(s_match.group(1))
            c_match = re.search(r'"?Critique"?\s*[:=]\s*"(.*?)"', text, re.DOTALL)
            if c_match: critique = c_match.group(1)
            else: critique = text.replace(json_cand, "")[:200] if json_cand != text else text[:200]

        return {
            "Persona": persona,
            "Score": max(0, min(10, score)),
            "Critique": critique
        }

    def evaluate(self, question: str, answer: str, manual_critique: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        
        for p in self.personas:
            prompt = (
                f"Role: {p.name}. Focus: {p.focus}. Instruction: {p.prompt_behavior}\n"
                f"Format: STRICTLY a single JSON object with keys 'Score' (0-10) and 'Critique' (string).\n"
                f"Do NOT include any extra text, introduction, or conversation outside the JSON object.\n\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n\n"
                f"JSON Output:"
            )
            raw = self.generator.generate(prompt, temperature=0.3)[0]
            parsed = self._parse_json_robust(raw, p.name)
            results.append(parsed)

        if manual_critique:
            results.append({
                "Persona": "Human User",
                "Score": 1,
                "Critique": manual_critique
            })
            
        return results

class DebateManager:
    def __init__(self, cfg: MirrorMindConfig):
        self.cfg = cfg
        self.generator = TextGenerator(cfg.model_name, device=cfg.device)
        self.embedder = Embedder(cfg.embedding_model, device=cfg.device)
        self.critics = CriticEnsemble(self.generator)

    def _best_of_n(self, question: str, n: int, temperature: float) -> str:
        """Tournament Selection: Generate N drafts, return the one most relevant to the Question."""
        console.log(f"[yellow]Running Best-of-{n} Tournament...[/yellow]")
        prompt = f"Question: {question}\nTask: Provide the most accurate answer. Output only the answer text.\nAnswer:"
        
        candidates = self.generator.generate(prompt, temperature=temperature, num_return=n)
        
        scores = []
        for cand in candidates:
            if not cand: continue
            relevance = self.embedder.get_score(question, cand)
            scores.append(relevance)
            
        best_idx = np.argmax(scores) if scores else 0
        console.log(f"[green]Winner Score: {scores[best_idx]:.4f}[/green]")
        return candidates[best_idx] if candidates else "Generation failed."

    def run(self, question: str, user_critique: Optional[str] = None, callback: Optional[Callable] = None):
        current_answer = self._best_of_n(question, self.cfg.num_candidates, self.cfg.initial_temp)
        session_data = {"rounds": []}

        for i in range(self.cfg.max_iters):
            critique_data = self.critics.evaluate(question, current_answer, user_critique if i == 0 else None)
            
            avg_score = np.mean([c["Score"] for c in critique_data])
            combined_text = "\n".join([f"[{c['Persona']}] Score {c['Score']}/10: {c['Critique']}" for c in critique_data])
            
            plan_prompt = (
                f"Role: Chief Strategist. Task: Create a concise, numbered Action Plan to fix the draft.\n"
                f"Instructions: Focus strictly on improving accuracy and clarity based on the feedback.\n\n"
                f"Question: {question}\nDraft: {current_answer}\nFeedback:\n{combined_text}\n\n"
                f"Action Plan (Numbered list only):"
            )
            plan = self.generator.generate(plan_prompt, temperature=0.2)[0]
            dynamic_temp = self.cfg.final_temp + (1.0 - (avg_score/10.0)) * 0.5
            rev_prompt = (
                f"Original Question: {question}\n"
                f"Current Draft: {current_answer}\n"
                f"Action Plan: {plan}\n\n"
                f"Task: Write the final, polished answer incorporating all fixes. Output ONLY the new answer text.\n"
                f"New Answer:"
            )
            new_answer = self.generator.generate(rev_prompt, temperature=dynamic_temp)[0]  
            sim = self.embedder.get_score(current_answer, new_answer)
            
            round_info = {
                "iteration": i + 1,
                "draft": current_answer,
                "critiques": critique_data,
                "plan": plan,
                "revised": new_answer,
                "avg_score": avg_score,
                "novelty": 1.0 - sim
            }
            session_data["rounds"].append(round_info)
            
            if callback and not callback(round_info): break

            if avg_score >= self.cfg.critique_threshold:
                console.log("[green]Excellent Quality Reached![/green]")
                current_answer = new_answer
                break
                
            if sim > self.cfg.similarity_threshold:
                console.log("[yellow]Stagnation (No semantic change).[/yellow]")
                current_answer = new_answer
                break
            
            current_answer = new_answer
            
        return session_data
