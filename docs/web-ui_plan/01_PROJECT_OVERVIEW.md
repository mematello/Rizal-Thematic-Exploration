# Project Overview: Rizal Thematic Exploration System

## Executive Summary

The **Rizal Thematic Exploration System** is a web-based semantic search engine for José Rizal's novels (*Noli Me Tangere* and *El Filibusterismo*). It uses hybrid AI analysis (lexical + semantic scoring) to help Filipino students and educators explore themes, characters, and passages with academic rigor.

---

## The Problem We're Solving

### Current State
Filipino students studying Rizal's works face several challenges:
- **Limited Access**: Physical books may be unavailable or expensive
- **Search Limitations**: Traditional Ctrl+F only finds exact word matches
- **Context Loss**: Difficult to find thematic connections across both novels
- **Mobile-First Reality**: 95% of Filipino students access educational content via smartphones

### Our Solution
A web application that:
1. **Understands Meaning**: Searches by concept, not just keywords
   - Query: "Edukasyon bilang pag-asa" finds relevant passages even if they use different words
2. **Compares Novels**: Side-by-side results from Noli and El Fili
3. **Mobile-Optimized**: Works flawlessly on low-end smartphones with limited bandwidth
4. **Academic Credibility**: Shows both semantic and lexical scores for transparency

---

## Core Features (MVP Scope)

### 1. Semantic Search
**User Input**: Natural language queries in Filipino/Tagalog
**System Output**: Ranked passages with dual scoring:
- **Semantic Score**: AI understanding of conceptual similarity
- **Lexical Score**: Traditional keyword matching

**Example**:
- Query: `"Katarungan para sa mahihirap"`
- Returns passages about justice for the poor, even if using synonyms like "hustisya" or "pagkakapantay-pantay"

### 2. Dual Novel Comparison
**Desktop**: Split-screen view (Noli left, El Fili right)
**Mobile**: Swipeable tabs with preserved scroll positions

### 3. Thematic Exploration
Pre-defined theme cards (e.g., Education, Justice, Love, Violence) that trigger curated searches.

### 4. Context Expansion
Click "Show Context" to see surrounding sentences (prev/next), helping users understand the broader narrative.

### 5. Educational Transparency
Visual score breakdowns teach students *why* a result was returned:
- Teal bar: How semantically relevant
- Amber bar: How many exact keywords matched

---

## Target Audience

### Primary Users
1. **High School Students** (Ages 15-18)
   - Preparing for required Rizal literature exams
   - Need quick, accurate passage retrieval for essays
   
2. **College Students** (Ages 18-22)
   - Conducting thematic analysis for research papers
   - Comparing character development across novels

3. **Educators**
   - Creating lesson plans
   - Finding examples for specific themes

### User Characteristics
- **Language**: Mix of English and Filipino proficiency
- **Device**: 95% mobile (primarily Android, low-to-mid range)
- **Connectivity**: Mobile data, often with bandwidth constraints
- **Technical Literacy**: Moderate; familiar with Google search, social media

---

## Success Metrics

### Performance Targets
- [ ] **Page Load**: < 3 seconds on 3G connection
- [ ] **Search Response**: < 2 seconds from query to results
- [ ] **Bundle Size**: < 500KB total (initial load)
- [ ] **Accessibility**: WCAG 2.1 AA compliance

### User Experience Goals
- [ ] **Mobile-First**: 100% feature parity on mobile
- [ ] **Clarity**: Students understand the scoring system without technical knowledge
- [ ] **Trust**: Results feel academically credible, not "AI magic"

### Usage Goals (Post-Launch)
- 100+ daily active users within 3 months
- Average session duration: 5+ minutes (indicates engagement)
- Return visit rate: 40%+ (indicates value)

---

## Out of Scope for MVP

The following features are explicitly **not** included in the initial release:

### Authentication & Personalization
- ❌ User accounts
- ❌ Saved searches
- ❌ Reading history
- ❌ Bookmarks/favorites

**Rationale**: Adds complexity (auth, database user tables, GDPR compliance) without proven demand. Can add post-MVP if users request it.

### Advanced Features
- ❌ PDF export of results
- ❌ Citation generator
- ❌ Social sharing
- ❌ Collaborative annotations

**Rationale**: Focus on core search functionality first. These are "nice-to-haves" that can be prioritized based on user feedback.

### Content Expansion
- ❌ Additional Rizal works (poetry, essays)
- ❌ Multi-language interface (English UI only; content stays Filipino)
- ❌ Audio narration

**Rationale**: Stick to the two canonical novels for MVP. Scope creep is the enemy of shipping.

---

## Design Philosophy

### "Paper & Ink" Aesthetic
The visual design evokes 19th-century manuscripts while maintaining modern usability:

- **Colors**: Warm cream backgrounds, brown/sepia tones, avoiding harsh whites
- **Typography**: Serif fonts for novel text (Crimson Text), sans-serif for UI (Roboto)
- **Texture**: Subtle CSS noise filters (not image-based) to suggest aged paper

### Academic Rigor
The interface prioritizes **credibility** over flashiness:
- No gamification elements
- Clear methodology explanations
- Transparency in scoring (users see both AI and traditional scores)
- Source attribution (chapter numbers, titles)

### Mobile-First Accessibility
Every design decision considers the 95% mobile user base:
- Touch targets ≥ 48px
- Text ≥ 18px for novel passages
- Offline capabilities (service worker caching)
- Bandwidth transparency (e.g., "Load More (~15KB)")

---

## Technical Constraints

### Performance Budget
- **Total Page Weight**: < 500KB (critical for mobile data)
  - Fonts: 60KB (subsetted)
  - JavaScript: 180KB (gzipped)
  - CSS: 12KB
  - Icons: 5KB (SVG)
  - Initial data: ~15KB

### Browser Support
- **Mobile**: Chrome/Safari (last 2 versions), iOS Safari
- **Desktop**: Chrome, Firefox, Safari, Edge (last 2 versions)
- **No**: Internet Explorer

### Offline Capability
- App shell cached via service worker
- Last 5 searches stored in localStorage
- "You are offline" state with helpful messaging (no hostile errors)

---

## Competitive Landscape

### Existing Solutions (and Their Limitations)

**1. Project Gutenberg**
- ✅ Free, full text
- ❌ No search by meaning (only Ctrl+F)
- ❌ No mobile optimization
- ❌ No thematic analysis

**2. Google Books**
- ✅ Full-text search
- ❌ Not semantic (keyword-only)
- ❌ Doesn't compare novels side-by-side
- ❌ Requires constant internet connection

**3. Academic Databases (JSTOR, etc.)**
- ✅ Scholarly articles about Rizal
- ❌ Not focused on the primary texts
- ❌ Paywalled
- ❌ Overwhelming for high school students

### Our Competitive Advantage
1. **Semantic Understanding**: Only tool that searches by meaning
2. **Novel Comparison**: Side-by-side results unique to this project
3. **Mobile-First**: Optimized for Filipino students' actual devices
4. **Free & Public**: No paywalls, no ads (thesis project, not commercial)

---

## Project Timeline

### Phase 1: Foundation (Weeks 1-2)
- Set up Next.js project
- Implement design system (Tailwind config, fonts)
- Create core components (ResultCard, SearchBar)

### Phase 2: Core Features (Weeks 3-6)
- Build search results page (desktop + mobile)
- Integrate with FastAPI backend
- Implement filtering and pagination

### Phase 3: Secondary Features (Weeks 7-8)
- Explore page with theme cards
- Methodology page (Student/Researcher tabs)
- Type-ahead suggestions

### Phase 4: Polish & Deploy (Weeks 9-10)
- Accessibility audit
- Performance optimization
- User testing with Filipino students
- Deploy to production (Vercel + Railway)

**Total Estimated Time**: 10-12 weeks (single full-time developer)

---

## Project Stakeholders

### Development Team
- **Marcus Oliver**: Full-stack developer (frontend + backend integration)
- **Antigravity**: Backend engine (RizalEngine, BERT embeddings)
- **Gemini**: Design system & UI/UX specifications

### Advisors
- Dr. Melvin Ballera (academic guidance)
- Filipino literature expert (content accuracy)
- Student beta testers (UX feedback)

---

## Risk Assessment

### Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| BERT embeddings too slow (>5s) | High | Medium | Pre-compute embeddings; use Redis caching |
| 500KB budget exceeded | Medium | Medium | Aggressive code splitting; font subsetting |
| Mobile performance on low-end devices | High | Medium | Test on real devices early; optimize bundle |

### Scope Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Feature creep (adding non-MVP features) | High | High | Strict adherence to "Out of Scope" list |
| Perfectionism (over-polishing MVP) | Medium | Medium | Set "good enough" threshold; ship iteratively |

### User Adoption Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Students don't understand semantic search | High | Low | Clear methodology page; visual score explanations |
| Results not academically credible | High | Low | Show both semantic + lexical; cite sources clearly |

---

## Next Steps

After reading this overview, proceed to:
1. **02_ARCHITECTURE.md**: Understand the system design
2. **03_TECH_STACK.md**: Review technology choices
3. **04_COMPONENT_SPECS.md**: Dive into component details
4. **05_API_INTEGRATION.md**: Learn backend integration
5. **06_IMPLEMENTATION_PLAN.md**: Follow step-by-step roadmap

---

## Questions & Clarifications

If you encounter ambiguities while implementing, refer to:
- **Design decisions**: DESIGN_SYSTEM.md
- **Component behavior**: 04_COMPONENT_SPECS.md
- **Technical constraints**: This document (Section: Technical Constraints)

For questions not covered in documentation, consult the project stakeholders or create a GitHub issue for discussion.