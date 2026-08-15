/**
 * The handoff's two open decisions, each reduced to a single switch.
 *
 * Both alternatives are real candidates rather than leftovers, so they are
 * kept reachable instead of being deleted. Changing either constant changes
 * the whole site — no component reads anything else.
 */

/**
 * `backyard` is the white/slate spec build. `techno` is the Techno Bowl
 * arcade palette; because every colour is a CSS variable, switching is the
 * `[data-theme='techno']` block in globals.css and nothing more.
 */
export const THEME: 'backyard' | 'techno' = 'backyard';

/**
 * Table header type. `pixel` is Press Start 2P at 8px (the spec default);
 * `plain` is Inter at 12px medium.
 */
export const TABLE_HEADER_FONT: 'pixel' | 'plain' = 'pixel';
