---
description: Complete UI tasks autonomously without requiring approval for each step
---

# UI Task Workflow

This workflow enables autonomous completion of UI development tasks.

// turbo-all

## Steps

1. **Understand the UI requirement**
   - Review the user's request
   - Identify which components need to be modified or created

2. **Make code changes**
   - Edit the necessary frontend files (components, pages, styles)
   - Follow existing patterns and design system

3. **Verify in browser**
   - Open the frontend in browser (http://localhost:3000)
   - Test the changes visually
   - Interact with the UI to ensure functionality works

4. **Fix any issues found**
   - If bugs or visual issues are discovered, fix them immediately
   - Re-test after each fix

5. **Confirm completion**
   - Verify all requirements are met
   - Report back to user with summary of changes

## Notes

- The `// turbo-all` annotation at the top means all commands in this workflow will auto-run
- This includes browser interactions, which will complete autonomously
- The agent will still show you what it's doing, but won't wait for approval at each step
