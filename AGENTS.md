# Working agreement

Keep each exercise in its own directory under exercises/. Preserve plans,
reproducible source/configuration, reviewed compact results, limitations, and
embedded PNG charts with SVG links. Never label a planned experiment as executed.
“Overlay” means putting compared methods on the same chart with shared axes.

The user has authorized persisting completed work to this repository. Review the
staged files, follow SECURITY.md, and run pre-commit/Gitleaks before commit and push.
Keep credentials, private host details, raw logs and traces, environments, weights
and compiler caches out of Git. The repository is private initially but must be
handled as if its contents were public.

Separate uninstrumented timing from profiling. Verify each algorithm against its
own reference; different normalization algorithms need not produce equal outputs.
Include training-specific activation/gradient costs and label inference controls.
